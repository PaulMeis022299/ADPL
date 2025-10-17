import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)

        # trnsformer layers for decoder
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=8,
                dim_feedforward=d_latent * 4,
                batch_first=True,  # input shape: (B, seq_len, d_model)
            ) for _ in range(6)
        ])

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(d_latent),
            torch.nn.Linear(d_latent, n_tokens)
            )
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x.clamp(0, self.n_tokens - 1).long()
        B, H, W = x.shape
        seq_len = H * W
        # flatten 2D patches to sequence
        x_seq = x.view(B, seq_len)  # (B, seq_len)
        # embed tokens
        x_emb = self.token_embedding(x_seq)  # (B, seq_len, d_latent)

        # shift input by 1 position with dummy at start
        dummy_token = torch.zeros(B, 1, self.d_latent, device=x.device)
        x_shift = torch.cat([dummy_token, x_emb[:, :-1, :]], dim=1)  # (B, seq_len, d_latent)
        # mask to prevent attention to future tokens using the Transformer helper
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(device=x.device)

        # pass through transformer layers
        h = x_shift
        for layer in self.layers:
            h = layer(h, src_mask=mask)

        logits = self.to_logits(h)  # (B, seq_len, n_tokens)
        logits = logits.view(B, H, W, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        self.eval()
        seq_len = h * w
        x = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        for i in range(seq_len):
            logits, _ = self.forward(x.view(B, h, w))
            logits = logits.view(B, seq_len, self.n_tokens)
            probs = torch.nn.functional.softmax(logits[:, i, :], dim=-1)
            x[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        return x.view(B, h, w)
