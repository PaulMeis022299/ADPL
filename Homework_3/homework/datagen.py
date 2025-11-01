

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json, numpy as np
    from .cot import CoTModel
    from .data import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    import torch
    from pathlib import Path

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
        questions = [q for q, _ in batch]
        true_answers = [a for _, a in batch]

        with torch.no_grad():
            outputs = cot_model.batched_generate(
                questions,
                num_return_sequences=oversample,
                temperature=temperature,
            )

        print(len(outputs))
        print(len(outputs))

        for i, (question, true_answer) in enumerate(zip(questions, true_answers)):
            completions = outputs[i]
            selected = None
            for completion in completions:
                try:
                    parsed = cot_model.parse_answer(completion)
                    #print("Raw: ", completion)
                    print("Parsed: ", parsed)
                    print("True: ", true_answer)
                    print('')
                    if parsed == true_answer:
                        selected = completion
                        break
                except (IndexError, ValueError):
                    continue

            if selected is not None:
                generated_data.append([dataset[idx + i][0], true_answer, selected])

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Saved {len(generated_data)} examples to {output_json}")


   

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
