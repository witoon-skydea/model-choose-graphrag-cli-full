"""
LLM module for RAG system with model selection
"""
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain

# Default LLM model
DEFAULT_LLM_MODEL = "llama3:8b"

# Available LLM models
AVAILABLE_LLM_MODELS = {
    "llama3": "llama3:8b",
    "llama3-70b": "llama3:70b",
    "phi3": "phi3:latest",
    "gemma": "gemma:7b",
    "mistral": "mistral:latest",
    "codellama": "codellama:latest",
    "wizardcoder": "wizardcoder:latest",
    "llava": "llava:latest"
}

# Define a prompt template that includes context
PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

User Question: {question}

Your Answer:
"""

def get_llm_model(model_name=None, temperature=0.7):
    """
    Get the LLM model with support for multiple models
    
    Args:
        model_name: Name of the model to use (uses default if None)
        temperature: Temperature setting for the model
    
    Returns:
        Ollama LLM model
    """
    if model_name is None:
        model = DEFAULT_LLM_MODEL
    elif model_name in AVAILABLE_LLM_MODELS:
        model = AVAILABLE_LLM_MODELS[model_name]
    else:
        # If the model_name is not in our predefined list, use it directly
        # This allows for using custom model names
        model = model_name
    
    return Ollama(model=model, temperature=temperature)

def format_context(documents: List[Document]) -> str:
    """
    Format a list of documents into a context string
    
    Args:
        documents: List of documents
        
    Returns:
        Formatted context string
    """
    return "\n\n".join([doc.page_content for doc in documents])

def generate_response(llm, documents: List[Document], question: str, custom_template=None) -> str:
    """
    Generate a response using the LLM and context documents
    
    Args:
        llm: LLM model
        documents: List of context documents
        question: User question
        custom_template: Custom prompt template (optional)
        
    Returns:
        Generated response
    """
    # Format documents into context string
    context = format_context(documents)
    
    # Create prompt from template
    template = custom_template if custom_template else PROMPT_TEMPLATE
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    response = chain.invoke({"context": context, "question": question})
    
    return response["text"]

def list_available_llm_models():
    """
    List all available LLM models
    
    Returns:
        Dictionary of available models
    """
    return AVAILABLE_LLM_MODELS
