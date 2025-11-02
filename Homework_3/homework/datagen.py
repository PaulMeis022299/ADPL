

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    from .cot import CoTModel
    from .data import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    import torch
    from pathlib import Path
    import math

    # Load instruction SmolLM2 model
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    # Wrap model in cot_model
    cot_model = CoTModel()
    cot_model.model = model.to("cuda")
    cot_model.tokenizer = tokenizer
    cot_model.model.eval()

    dataset = Dataset("train")
    batch_size = 16

    generated_data = []

    for idx in tqdm(range(0, len(dataset), batch_size), desc="Generating dataset"):
        batch = dataset[idx : idx + batch_size]
        questions = [cot_model.format_prompt(q) for q, _ in batch]
        true_answers = [a for _, a in batch]

        with torch.no_grad():
            outputs = cot_model.batched_generate(
                questions,
                num_return_sequences=oversample,
                temperature=temperature,
            )

        for i, (question, true_answer) in enumerate(zip(questions, true_answers)):
            completions = outputs[i]
            selected = None
            for completion in completions:
              try:
                parsed = cot_model.parse_answer(completion)
                # --- DEBUG
                #if i < 5:
                #  print(f"\n🧩 Example {i + 1}")
                #  print(f"Q: {dataset[idx + i][0]}")
                #  print(f"True: {true_answer:.4g}")

                #  for j, c in enumerate(completions[:3]):
                #    parsed_val = None
                #    try:
                #      parsed_val = cot_model.parse_answer(c)
                #    except Exception:
                #      pass
                #    preview = c.strip().replace("\n", " ")[:120]
                #    print(f"  [{j}] parsed={parsed_val} | {preview}...")
                # --- DEBUG
                if not math.isnan(parsed): 
                  if abs(parsed - true_answer) / max(abs(true_answer), 1e-6) < 0.05:
                    selected = completion
                    generated_data.append([dataset[idx + i][0], true_answer, selected])
                  break
              except (IndexError, ValueError):
                continue
          

        torch.cuda.empty_cache()

    #generated_data = [
    #[q, a, r]
    #for (q, a, r) in generated_data
    #if r is not None and isinstance(r, str) and "<answer>" in r
    #]
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Saved {len(generated_data)} examples to {output_json}")


   

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
