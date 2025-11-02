from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        
        messages = [
        {"role": "system", "content": ("You are a precise reasoning assistant that converts between measurement units. "
                                        "Carefully compute the conversion and return the result as a pure float, "
                                        "wrapped in <answer></answer> tags. Be concise and show brief reasoning.")
        },
        {"role": "user", "content": "How many feet are there in 2 meters?"},
        {"role": "assistant", "content": "1 meter = 3.28084 feet. 2 * 3.28084 = 6.56168. <answer>6.56168</answer>"},
        {"role": "user", "content": "How much does 5 pounds weigh in kg?"},
        {"role": "assistant", "content": "1 pound = 0.453592 kg. 5 * 0.453592 = 2.26796. <answer>2.26796</answer>"},
        {"role": "user", "content": "Convert 3 meters into centimeters."},
        {"role": "assistant", "content": "1 meter = 100 centimeters. 3 × 100 = <answer>300</answer>"},
        {"role": "user", "content": question},
        ]

        # Add the target question
        messages.append({"role": "user", "content": f"Question: {question}"})
        
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
