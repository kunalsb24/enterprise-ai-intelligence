import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """Generate text using a local Hugging Face language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response from chat messages."""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()