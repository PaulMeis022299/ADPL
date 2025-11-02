from .base_llm import BaseLLM
from .sft import test_model
from .sft import tokenize, format_example
from .data import Dataset, benchmark

from pathlib import Path
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import Trainer, TrainingArguments
import torch


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(prompt: str, answer: str, reasoning: str) -> dict[str, str]:
    """
    Construct a question / answerm + reasoning pair
    """
    answer_with_reasoning = reasoning.strip()
    return {"question": prompt.strip(), "answer": answer_with_reasoning}


class TokenizedDataset:
    def __init__(self, tokenizer, rft_list: list[tuple], format_fn):
        """
        rft_list: list of [question:str, float_answer:float, reasoning_with_answer:str]
        format_fn: function(q, float_answer, reasoning_with_answer) -> {'question': str, 'answer': str}
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = rft_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, true_ans, reasoning = self.data[idx]
        formatted = self.format_fn(q, true_ans, reasoning)
        return tokenize(self.tokenizer, formatted["question"], formatted["answer"])




def train_model(
    output_dir: str,
    **kwargs,
):
   
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer

    lora_config = LoraConfig(
        r=16,                        
        lora_alpha=64,              
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    if torch.cuda.is_available():
        model.enable_input_require_grads()

    train_data = Dataset("rft")

    train_dataset = TokenizedDataset(tokenizer, train_data, format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        save_strategy="epoch",
        save_total_limit= 2,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
