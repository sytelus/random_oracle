import re
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM


class LocalHFLLM(BaseLLM):
    """Simple local Hugging Face model wrapper for offline experimentation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        num_workers: int = 1,
        strict_json: bool = False,
    ):
        super().__init__(model_name, config, num_workers, strict_json)

        self.device = torch.device(config.get("device", "cpu"))
        self.max_new_tokens = int(config.get("max_tokens", 128))
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 1.0))
        self.repetition_penalty = float(config.get("repetition_penalty", 1.0))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        segments = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            segments.append(f"{role}: {content}")
        segments.append("ASSISTANT:")
        return "\n".join(segments)

    def _generate_text(self, prompt_text: str) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        prompt_text = self._messages_to_text(messages)
        return self._generate_text(prompt_text)

    def _extract_probability_field(self, schema: Dict[str, Any]) -> str:
        try:
            properties = schema["json_schema"]["schema"]["properties"]["responses"]["items"][
                "properties"
            ]
            for candidate in ("probability", "confidence", "perplexity", "nll"):
                if candidate in properties:
                    return candidate
        except Exception:
            pass
        return "probability"

    def _infer_num_samples(self, messages: List[Dict[str, str]]) -> int:
        for message in messages:
            content = message.get("content", "")
            match = re.search(r"Generate\s+(\d+)\s+responses", content, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return int(self.config.get("num_samples", 5))

    def _chat_with_format(self, messages: List[Dict[str, str]], schema) -> Dict[str, Any]:
        num_samples = max(1, self._infer_num_samples(messages))
        field_name = self._extract_probability_field(schema)

        responses = []
        for _ in range(num_samples):
            text = self._chat(messages)
            responses.append({"text": text, field_name: 1.0 / num_samples})

        return {"responses": responses}
