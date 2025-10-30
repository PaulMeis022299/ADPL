from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
    
        # 2 examples
        examples = [
            {
                "question": "How many feet are there in 2 meters?",
                "answer": (
                    "1 meter = 3.28084 feet. "
                    "2 * 3.28084 = 6.56168. "
                    "Final answer: <answer>6.56168</answer>"
                ),
            },
            {
                "question": "How much does 5 pounds weigh in kilograms?",
                "answer": (
                    "1 pound = 0.453592 kilograms. "
                    "5 * 0.453592 = 2.26796. "
                    "Final answer: <answer>2.26796</answer>"
                ),
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise reasoning assistant that performs unit conversions. "
                    "Carefully compute the conversion and return the result as a pure float "
                    "wrapped in <answer></answer> tags. Be concise."
                ),
            },
        ]

        for ex in examples:
            messages.append({"role": "user", "content": f"Question: {ex['Question']}"})
            messages.append({"role": "assistant", "content": ex["Answer"]})

        # Add the target question
        messages.append({"role": "user", "content": f"Question: {question}"})
        

        # Apply the SmolLM2 chat template
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
