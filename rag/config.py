"""
Configuration module for multi-tenant RAG system with model selection
"""
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Constants
DEFAULT_CONFIG_FILE = "companies/config.json"
DEFAULT_COMPANY = "default"
DEFAULT_LLM_MODEL = "llama3:8b"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large:latest"

class SystemConfig:
    """
    Class to manage global system configuration
    """
    def __init__(self, config_path: str = "companies/system_config.json"):
        """
        Initialize the system configuration
        
        Args:
            config_path: Path to the config file
        """
        self.config_path = config_path
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file or create default if not exists
        
        Returns:
            Dict containing configuration
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading system config: {e}. Creating default config.")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """
        Create default system configuration
        
        Returns:
            Dict containing default configuration
        """
        default_config = {
            "default_llm_model": DEFAULT_LLM_MODEL,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "temperature": 0.7,
            "top_k": 4,
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default config
        self._save_config(default_config)
        
        return default_config
    
    def _save_config(self, config_data: Optional[Dict] = None) -> None:
        """
        Save configuration to file
        
        Args:
            config_data: Configuration data to save (uses self.config_data if None)
        """
        if config_data is None:
            config_data = self.config_data
            
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def get_default_llm_model(self) -> str:
        """
        Get the default LLM model
        
        Returns:
            Default LLM model name
        """
        return self.config_data.get("default_llm_model", DEFAULT_LLM_MODEL)
    
    def set_default_llm_model(self, model_name: str) -> None:
        """
        Set the default LLM model
        
        Args:
            model_name: Model name to set as default
        """
        self.config_data["default_llm_model"] = model_name
        self._save_config()
    
    def get_default_embedding_model(self) -> str:
        """
        Get the default embedding model
        
        Returns:
            Default embedding model name
        """
        return self.config_data.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)
    
    def set_default_embedding_model(self, model_name: str) -> None:
        """
        Set the default embedding model
        
        Args:
            model_name: Model name to set as default
        """
        self.config_data["default_embedding_model"] = model_name
        self._save_config()
    
    def get_temperature(self) -> float:
        """
        Get the default temperature
        
        Returns:
            Default temperature
        """
        return float(self.config_data.get("temperature", 0.7))
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the default temperature
        
        Args:
            temperature: Temperature to set as default
        """
        self.config_data["temperature"] = float(temperature)
        self._save_config()

    def get_top_k(self) -> int:
        """
        Get the default number of chunks to return
        
        Returns:
            Default top_k
        """
        return int(self.config_data.get("top_k", 4))
    
    def set_top_k(self, top_k: int) -> None:
        """
        Set the default number of chunks to return
        
        Args:
            top_k: Top_k to set as default
        """
        self.config_data["top_k"] = int(top_k)
        self._save_config()
    
    def get_chunk_settings(self) -> Dict:
        """
        Get chunk size and overlap settings
        
        Returns:
            Dict with chunk_size and chunk_overlap
        """
        return {
            "chunk_size": self.config_data.get("chunk_size", 1000),
            "chunk_overlap": self.config_data.get("chunk_overlap", 200)
        }
    
    def set_chunk_settings(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Set chunk size and overlap settings
        
        Args:
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
        """
        self.config_data["chunk_size"] = int(chunk_size)
        self.config_data["chunk_overlap"] = int(chunk_overlap)
        self._save_config()
    
    def get_all_settings(self) -> Dict:
        """
        Get all system settings
        
        Returns:
            Dict with all settings
        """
        return self.config_data


class CompanyConfig:
    """
    Class to manage company configurations for the multi-tenant RAG system
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE):
        """
        Initialize the company configuration
        
        Args:
            config_path: Path to the config file
        """
        self.config_path = config_path
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file or create default if not exists
        
        Returns:
            Dict containing configuration
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}. Creating default config.")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """
        Create default configuration
        
        Returns:
            Dict containing default configuration
        """
        # Get system config for default model settings
        sys_config = SystemConfig()
        default_llm = sys_config.get_default_llm_model()
        default_embedding = sys_config.get_default_embedding_model()
        
        default_config = {
            "companies": {
                DEFAULT_COMPANY: {
                    "name": "Default Company",
                    "description": "Default company for RAG system",
                    "db_dir": f"db/{DEFAULT_COMPANY}",
                    "llm_model": default_llm,
                    "embedding_model": default_embedding
                }
            },
            "active_company": DEFAULT_COMPANY
        }
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default config
        self._save_config(default_config)
        
        return default_config
    
    def _save_config(self, config_data: Optional[Dict] = None) -> None:
        """
        Save configuration to file
        
        Args:
            config_data: Configuration data to save (uses self.config_data if None)
        """
        if config_data is None:
            config_data = self.config_data
            
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def get_companies(self) -> List[str]:
        """
        Get list of company IDs
        
        Returns:
            List of company IDs
        """
        return list(self.config_data.get("companies", {}).keys())
    
    def get_company_details(self, company_id: str) -> Dict:
        """
        Get details for a specific company
        
        Args:
            company_id: ID of the company
            
        Returns:
            Dict containing company details
            
        Raises:
            ValueError: If company_id does not exist
        """
        companies = self.config_data.get("companies", {})
        if company_id not in companies:
            raise ValueError(f"Company ID '{company_id}' not found")
        
        return companies[company_id]
    
    def add_company(self, company_id: str, name: str, description: str = "", 
                   llm_model: Optional[str] = None, embedding_model: Optional[str] = None) -> None:
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
        if company_id in self.config_data.get("companies", {}):
            raise ValueError(f"Company ID '{company_id}' already exists")
        
        # Get system defaults if not specified
        if llm_model is None or embedding_model is None:
            sys_config = SystemConfig()
            
        if llm_model is None:
            llm_model = sys_config.get_default_llm_model()
            
        if embedding_model is None:
            embedding_model = sys_config.get_default_embedding_model()
        
        # Create company entry
        self.config_data.setdefault("companies", {})[company_id] = {
            "name": name,
            "description": description,
            "db_dir": f"db/{company_id}",
            "llm_model": llm_model,
            "embedding_model": embedding_model
        }
        
        # Create company database directory
        db_dir = self.config_data["companies"][company_id]["db_dir"]
        os.makedirs(db_dir, exist_ok=True)
        
        # Save config
        self._save_config()
        
    def remove_company(self, company_id: str) -> None:
        """
        Remove a company
        
        Args:
            company_id: ID of the company to remove
            
        Raises:
            ValueError: If company_id does not exist or is the active company
        """
        if company_id not in self.config_data.get("companies", {}):
            raise ValueError(f"Company ID '{company_id}' not found")
        
        if company_id == self.config_data.get("active_company"):
            raise ValueError(f"Cannot remove active company '{company_id}'. Switch to another company first.")
        
        # Remove company
        del self.config_data["companies"][company_id]
        
        # Save config
        self._save_config()
    
    def set_active_company(self, company_id: str) -> None:
        """
        Set the active company
        
        Args:
            company_id: ID of the company to set as active
            
        Raises:
            ValueError: If company_id does not exist
        """
        if company_id not in self.config_data.get("companies", {}):
            raise ValueError(f"Company ID '{company_id}' not found")
        
        # Set active company
        self.config_data["active_company"] = company_id
        
        # Save config
        self._save_config()
        
    def get_active_company(self) -> str:
        """
        Get the active company ID
        
        Returns:
            ID of the active company
        """
        return self.config_data.get("active_company", DEFAULT_COMPANY)
    
    def get_active_company_details(self) -> Dict:
        """
        Get details for the active company
        
        Returns:
            Dict containing active company details
        """
        active_company = self.get_active_company()
        return self.get_company_details(active_company)
    
    def get_db_path(self, company_id: Optional[str] = None) -> str:
        """
        Get database path for a company
        
        Args:
            company_id: ID of the company (uses active company if None)
            
        Returns:
            Path to the company's database directory
        """
        if company_id is None:
            company_id = self.get_active_company()
        
        company_details = self.get_company_details(company_id)
        db_dir = company_details.get("db_dir", f"db/{company_id}")
        
        # Ensure directory exists
        os.makedirs(db_dir, exist_ok=True)
        
        return db_dir
    
    def get_company_model_settings(self, company_id: Optional[str] = None) -> Dict:
        """
        Get model settings for a company
        
        Args:
            company_id: ID of the company (uses active company if None)
            
        Returns:
            Dict with llm_model and embedding_model
        """
        if company_id is None:
            company_id = self.get_active_company()
        
        company_details = self.get_company_details(company_id)
        
        # Get system defaults
        sys_config = SystemConfig()
        default_llm = sys_config.get_default_llm_model()
        default_embedding = sys_config.get_default_embedding_model()
        
        return {
            "llm_model": company_details.get("llm_model", default_llm),
            "embedding_model": company_details.get("embedding_model", default_embedding)
        }
    
    def set_company_model_settings(self, company_id: str, llm_model: Optional[str] = None, 
                                 embedding_model: Optional[str] = None) -> None:
        """
        Set model settings for a company
        
        Args:
            company_id: ID of the company
            llm_model: Custom LLM model for this company (None to leave unchanged)
            embedding_model: Custom embedding model for this company (None to leave unchanged)
            
        Raises:
            ValueError: If company_id does not exist
        """
        if company_id not in self.config_data.get("companies", {}):
            raise ValueError(f"Company ID '{company_id}' not found")
        
        # Update only provided values
        if llm_model is not None:
            self.config_data["companies"][company_id]["llm_model"] = llm_model
            
        if embedding_model is not None:
            self.config_data["companies"][company_id]["embedding_model"] = embedding_model
        
        # Save config
        self._save_config()
