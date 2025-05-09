# src/custom_llm.py

from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation
from huggingface_hub import InferenceClient
from typing import List
from pydantic import Field

class HuggingFaceInferenceLLM(LLM):
    model: str = Field(default="google/flan-t5-base")

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        client = InferenceClient(model=self.model)
        response = client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.5,
        )
        return response.strip()

    def _generate(
        self,
        prompts: List[str],
        stop: List[str] = None,
    ) -> List[Generation]:
        return [Generation(text=self._call(prompt, stop)) for prompt in prompts]

