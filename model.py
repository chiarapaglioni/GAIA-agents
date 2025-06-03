import os
from typing import Any, Callable
from smolagents import HfApiModel, InferenceClientModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalTransformersModel:
    def __init__(self, model_id: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, prompt: str, **kwargs):
        outputs = self.pipeline(prompt, **kwargs)
        return outputs[0]["generated_text"]

def get_huggingface_api_model(model_id: str, **kwargs) -> HfApiModel:
    """
    Returns a Hugging Face API model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        HfApiModel: Hugging Face API model instance.
    """
    return HfApiModel(model_id=model_id, token=os.getenv("HF_TOKEN"), **kwargs)


def get_inference_client_model(model_id: str, **kwargs) -> InferenceClientModel:
    """
    Returns an Inference Client model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        InferenceClientModel: Inference client model instance.
    """
    return InferenceClientModel(model_id=model_id, token=os.getenv("HF_TOKEN"), **kwargs)


def get_local_model(model_id: str, **kwargs) -> LocalTransformersModel:
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
        "HfApiModel": get_huggingface_api_model,
        "InferenceClientModel": get_inference_client_model,
        "LocalTransformersModel": get_local_model,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](model_id, **kwargs)