"""
Embeddings module for RAG system with model selection
"""
from langchain_community.embeddings import OllamaEmbeddings

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = {
    "mxbai": "mxbai-embed-large:latest",
    "nomic": "nomic-embed-text:latest",
    "llama3": "llama3:8b",  # Can use LLM models for embeddings too
    "phi3": "phi3:latest"
}

def get_embeddings_model(model_name=None):
    """
    Get the embeddings model with support for multiple models
    
    Args:
        model_name: Name of the model to use (uses default if None)
    
    Returns:
        OllamaEmbeddings model
    """
    if model_name is None:
        model = DEFAULT_EMBEDDING_MODEL
    elif model_name in AVAILABLE_EMBEDDING_MODELS:
        model = AVAILABLE_EMBEDDING_MODELS[model_name]
    else:
        # If the model_name is not in our predefined list, use it directly
        # This allows for using custom model names
        model = model_name
    
    return OllamaEmbeddings(model=model)

def list_available_embedding_models():
    """
    List all available embedding models
    
    Returns:
        Dictionary of available models
    """
    return AVAILABLE_EMBEDDING_MODELS
