import os
from typing import Any, Callable

from smolagents import HfApiModel, InferenceClientModel, LiteLLMModel, OpenAIServerModel


def get_huggingface_api_model(model_id: str, **kwargs) -> HfApiModel:
    """
    Returns a Hugging Face API model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        HfApiModel: Hugging Face API model instance.
    """
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    return HfApiModel(model_id=model_id, token=api_key, **kwargs)


def get_inference_client_model(model_id: str, **kwargs) -> InferenceClientModel:
    """
    Returns an Inference Client model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        InferenceClientModel: Inference client model instance.
    """
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    return InferenceClientModel(model_id=model_id, token=api_key, **kwargs)


def get_openai_server_model(model_id: str, **kwargs) -> OpenAIServerModel:
    """
    Returns an OpenAI server model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        OpenAIServerModel: OpenAI server model instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    api_base = os.getenv("OPENAI_API_BASE")
    if not api_base:
        raise ValueError("OPENAI_API_BASE is not set")

    return OpenAIServerModel(
        model_id=model_id, api_key=api_key, api_base=api_base, **kwargs
    )


def get_lite_llm_model(model_id: str, **kwargs) -> LiteLLMModel:
    """
    Returns a LiteLLM model instance.

    Args:
        model_id (str): The model identifier.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        LiteLLMModel: LiteLLM model instance.
    """
    return LiteLLMModel(model_id=model_id, **kwargs)


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
        "OpenAIServerModel": get_openai_server_model,
        "LiteLLMModel": get_lite_llm_model,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](model_id, **kwargs)