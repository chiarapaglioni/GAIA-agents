import os
from typing import Any, Callable

from smolagents import LiteLLMModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from functools import lru_cache


class LocalTransformersModel:
    def __init__(self, model_id: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, prompt: str, **kwargs):
        outputs = self.pipeline(prompt, **kwargs)
        return outputs[0]["generated_text"]

@lru_cache(maxsize=1)
def get_lite_llm_model(model_id: str,  **kwargs) -> LiteLLMModel:
    """
    Returns a LiteLLM model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        LiteLLMModel: LiteLLM model instance.
    """
    return LiteLLMModel(model_id=model_id, api_key=os.getenv("GEMINI_API"), **kwargs)


@lru_cache(maxsize=1)
def get_local_model(model_id: str, **kwargs) -> LocalTransformersModel:
    """
    Returns a Local Transformer model.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        LocalTransformersModel: LiteLLM model instance.
    """
    return LocalTransformersModel(model_id=model_id, **kwargs)


def get_model(model_type: str, model_id: str, **kwargs) -> Any:
    """
    Returns a model instance based on the specified type.

    Args:
        model_type (str): The type of the model (e.g., 'HfApiModel').
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        Any: Model instance of the specified type.
    """
    models: dict[str, Callable[..., Any]] = {
        "LiteLLMModel": get_lite_llm_model,
        "LocalTransformersModel": get_local_model,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](model_id, **kwargs)