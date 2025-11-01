

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json, numpy as np
    from .cot import CoTModel
    from .data import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load instruction SmolLM2 model
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    # Wrap model in cot_model
    cot_model = CoTModel()
    cot_model.model = model.to("cuda")
    cot_model.tokenizer = tokenizer
    cot_model.model.eval()

    # Load dataset
    data = Dataset("train")
    successful = []
    i = 0
    for q, true_answer in data:
        prompt = cot_model.format_prompt(q)

        outputs = cot_model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )
        
        flat_outputs = outputs[0]

        found_any = False

        for out in flat_outputs:
          parsed = cot_model.parse_answer(out)
          if np.isfinite(parsed) and abs(parsed - true_answer) / (abs(true_answer) + 1e-6) < 1e-3:
            successful.append([q, true_answer, out])
            found_any = True
            i += 1
            print("Sample ", i, " out of ", len(data),":")
            to_print = [q, true_answer, out]
            print(*to_print, sep='\n')
            print("")
            break


    with open(output_json, "w") as f:
        json.dump(successful, f, indent=2)

    print(f"Saved {len(successful)} / {len(data)} examples to {output_json}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
