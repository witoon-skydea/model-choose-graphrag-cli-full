"""
Vector store module for multi-tenant RAG system with model selection
"""
import os
from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .embeddings import get_embeddings_model, list_available_embedding_models
from .config import CompanyConfig, SystemConfig

def get_vector_store(persist_directory: Optional[str] = None, 
                    company_id: Optional[str] = None,
                    embedding_model: Optional[str] = None):
    """
    Get the vector store for a specific company with specified embedding model
    
    Args:
        persist_directory: Directory to persist the vector store (overrides company config)
        company_id: ID of the company (uses active company if None)
        embedding_model: Name of the embedding model to use (uses company setting if None)
        
    Returns:
        Chroma vector store
    """
    # If persist_directory is not provided, use company config
    if persist_directory is None:
        company_config = CompanyConfig()
        persist_directory = company_config.get_db_path(company_id)
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Determine embedding model
    if embedding_model is None and company_id is not None:
        # Get model from company config
        company_config = CompanyConfig()
        model_settings = company_config.get_company_model_settings(company_id)
        embedding_model = model_settings.get("embedding_model")
    
    # Get embeddings model
    embeddings = get_embeddings_model(embedding_model)
    
    # Return vector store
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def add_documents(vector_store, documents: List[Document]):
    """
    Add documents to the vector store
    
    Args:
        vector_store: Vector store to add documents to
        documents: List of documents to add
        
    Returns:
        None
    """
    vector_store.add_documents(documents)
    
def similarity_search(vector_store, query: str, k: int = 4):
    """
    Search for similar documents in the vector store
    
    Args:
        vector_store: Vector store to search
        query: Query to search for
        k: Number of results to return
        
    Returns:
        List of documents
    """
    return vector_store.similarity_search(query, k=k)

def list_companies():
    """
    List all available companies
    
    Returns:
        List of company IDs and their details
    """
    config = CompanyConfig()
    companies = config.get_companies()
    
    company_details = []
    for company_id in companies:
        details = config.get_company_details(company_id)
        company_details.append({
            "id": company_id,
            "name": details.get("name", company_id),
            "description": details.get("description", ""),
            "llm_model": details.get("llm_model", "default"),
            "embedding_model": details.get("embedding_model", "default"),
            "active": company_id == config.get_active_company()
        })
    
    return company_details

def get_active_company():
    """
    Get the active company
    
    Returns:
        Dict with active company details
    """
    config = CompanyConfig()
    active_company_id = config.get_active_company()
    details = config.get_company_details(active_company_id)
    
    return {
        "id": active_company_id,
        "name": details.get("name", active_company_id),
        "description": details.get("description", ""),
        "db_dir": details.get("db_dir", f"db/{active_company_id}"),
        "llm_model": details.get("llm_model", "default"),
        "embedding_model": details.get("embedding_model", "default")
    }

def add_company(company_id: str, name: str, description: str = "", 
               llm_model: Optional[str] = None, embedding_model: Optional[str] = None):
    """
    Add a new company
    
    Args:
        company_id: ID for the new company
        name: Name of the company
        description: Description of the company
        llm_model: Custom LLM model for this company
        embedding_model: Custom embedding model for this company
        
    Raises:
        ValueError: If company_id already exists
    """
    config = CompanyConfig()
    config.add_company(company_id, name, description, llm_model, embedding_model)

def remove_company(company_id: str):
    """
    Remove a company
    
    Args:
        company_id: ID of the company to remove
        
    Raises:
        ValueError: If company_id does not exist or is the active company
    """
    config = CompanyConfig()
    config.remove_company(company_id)

def set_active_company(company_id: str):
    """
    Set the active company
    
    Args:
        company_id: ID of the company to set as active
        
    Raises:
        ValueError: If company_id does not exist
    """
    config = CompanyConfig()
    config.set_active_company(company_id)

def set_company_models(company_id: str, llm_model: Optional[str] = None, 
                     embedding_model: Optional[str] = None):
    """
    Set models for a company
    
    Args:
        company_id: ID of the company
        llm_model: Custom LLM model for this company
        embedding_model: Custom embedding model for this company
        
    Raises:
        ValueError: If company_id does not exist
    """
    config = CompanyConfig()
    config.set_company_model_settings(company_id, llm_model, embedding_model)

def get_system_settings():
    """
    Get global system settings
    
    Returns:
        Dict with system settings
    """
    config = SystemConfig()
    return config.get_all_settings()

def set_system_settings(settings: Dict):
    """
    Update system settings
    
    Args:
        settings: Dict with settings to update
        
    Returns:
        None
    """
    config = SystemConfig()
    
    # Update LLM model
    if "default_llm_model" in settings:
        config.set_default_llm_model(settings["default_llm_model"])
    
    # Update embedding model
    if "default_embedding_model" in settings:
        config.set_default_embedding_model(settings["default_embedding_model"])
    
    # Update temperature
    if "temperature" in settings:
        config.set_temperature(float(settings["temperature"]))
    
    # Update top_k
    if "top_k" in settings:
        config.set_top_k(int(settings["top_k"]))
    
    # Update chunk settings
    if "chunk_size" in settings and "chunk_overlap" in settings:
        config.set_chunk_settings(
            int(settings["chunk_size"]),
            int(settings["chunk_overlap"])
        )
