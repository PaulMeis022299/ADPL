'''
+------------------+
|   Raw Images      |
|  (B, H=100, W=150, C=3) |
+------------------+
          |
          v
+------------------+
| Patch Extraction |
| (PatchAutoEncoder) |
| Output: Embeddings |
| (B, h=20, w=30, latent_dim=128) |
+------------------+
          |
          v
+------------------+
| BSQ Quantization |
| - L2 normalize   |
| - diff_sign      |
| Output: Binary Codes |
| (B, h=20, w=30, codebook_bits=10) |
+------------------+
          |
          v
+------------------+
| Tokenized Dataset |
| Convert Binary Codes → Integer Tokens |
| (B, h=20, w=30)  |
+------------------+
          |
          v
+----------------------------+
| Autoregressive Transformer |
| Input: Flattened Token Sequence |
| Output: Next-token Probabilities |
| (B, h, w, n_tokens=1024)  |
+----------------------------+
          |
          v
+------------------+
| Generated Token  |
| Sequence (sampled) |
| (B, h=20, w=30)  |
+------------------+
          |
          v
+------------------+
| Reconstruction   |
| - Tokens → BSQ Codes |
| - Decode Embeddings → Patches → Full Image |
| Output: Generated Images |
| (B, H=100, W=150, C=3) |
+------------------+
'''