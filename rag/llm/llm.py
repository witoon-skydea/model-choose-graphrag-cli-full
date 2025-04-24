"""
LLM module for GraphRAG system with model selection
"""
import json
from typing import List, Dict, Any, Tuple
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

# Enhanced prompt template that includes knowledge graph information
GRAPH_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following pieces of context and knowledge graph information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context from documents:
{context}

Knowledge Graph Information:
{graph_context}

User Question: {question}

Your Answer:
"""

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """
You are an expert in extracting entities and relationships from text.
Analyze the following text and extract:
1. All important entities (people, organizations, locations, concepts, etc.)
2. Relationships between these entities

Format your response as JSON:
```json
{
  "entities": [
    {"id": "unique_id_1", "name": "Entity Name 1", "type": "person|organization|location|concept|other", "attributes": {"key1": "value1"}},
    {"id": "unique_id_2", "name": "Entity Name 2", "type": "person|organization|location|concept|other", "attributes": {"key1": "value1"}}
  ],
  "relations": [
    {"source": "unique_id_1", "target": "unique_id_2", "type": "relationship_type", "attributes": {"key1": "value1"}}
  ]
}
```

IMPORTANT NOTES:
- Generate unique IDs that are short but descriptive (e.g., "john_doe_1", "apple_inc", "new_york_city")
- Only extract definitely stated relationships, don't infer tenuous connections
- Be specific with relationship types (e.g., "works_for", "located_in", "published", "founded_by")
- Include source and page number in attributes when available

Text to analyze:
{text}

JSON response:
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
    return "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
                       for i, doc in enumerate(documents)])

def format_graph_context(graph_data: List[Dict[str, Any]]) -> str:
    """
    Format graph data into a readable context string
    
    Args:
        graph_data: List of graph elements (entities and relationships)
        
    Returns:
        Formatted graph context string
    """
    if not graph_data:
        return "No relevant graph information available."
    
    formatted_entries = []
    for entry in graph_data:
        if entry["type"] == "entity":
            formatted_entries.append(f"Entity: {entry['name']} (Type: {entry['entity_type']})")
        elif entry["type"] == "relation":
            formatted_entries.append(f"Relation: {entry['source_name']} --[{entry['relation_type']}]--> {entry['target_name']}")
    
    return "\n".join(formatted_entries)

def generate_response(llm, documents: List[Document], question: str, custom_template=None, graph_data: List[Dict[str, Any]] = None) -> str:
    """
    Generate a response using the LLM, context documents, and knowledge graph
    
    Args:
        llm: LLM model
        documents: List of context documents
        question: User question
        custom_template: Custom prompt template (optional)
        graph_data: Knowledge graph data related to the question (optional)
        
    Returns:
        Generated response
    """
    # Format documents into context string
    context = format_context(documents)
    
    if graph_data:
        # Format graph data and use the graph prompt template
        graph_context = format_graph_context(graph_data)
        
        if custom_template:
            # If a custom template is provided and we have graph data,
            # try to add graph context to it if it has the right placeholder
            if "{graph_context}" in custom_template:
                template = custom_template
            else:
                # Otherwise use the default graph template
                template = GRAPH_PROMPT_TEMPLATE
        else:
            # Use default graph template
            template = GRAPH_PROMPT_TEMPLATE
            
        # Create prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "graph_context", "question"]
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        response = chain.invoke({
            "context": context, 
            "graph_context": graph_context, 
            "question": question
        })
    else:
        # Use standard template without graph data
        template = custom_template if custom_template else PROMPT_TEMPLATE
        
        # Create prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        response = chain.invoke({"context": context, "question": question})
    
    return response["text"]

def extract_entities_relations(llm, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relations from text using LLM
    
    Args:
        llm: LLM model
        text: Text to analyze
        
    Returns:
        Tuple of (entities, relations)
    """
    # Create a direct prompt for better JSON generation
    direct_prompt = f"""Extract entities and relationships from the following text.
Return a valid JSON object with 'entities' and 'relations' arrays.
Format your entire response as a valid JSON object only, nothing else.

For example:
{{
  "entities": [
    {{"id": "person_1", "name": "John Smith", "type": "person"}},
    {{"id": "org_1", "name": "Acme Corp", "type": "organization"}}
  ],
  "relations": [
    {{"source": "person_1", "target": "org_1", "type": "works_for"}}
  ]
}}

Text to analyze:
{text}

JSON response:"""
    
    # Get raw response from LLM
    response = llm.invoke(direct_prompt)
    result_text = response
    
    # Extract JSON from the response
    try:
        # Try to find JSON between curly braces
        if "{" in result_text and "}" in result_text:
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            json_str = result_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Extract entities and relations
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            
            # Add default IDs if missing
            for i, entity in enumerate(entities):
                if "id" not in entity and "name" in entity:
                    # Create a simple ID from the name
                    name = entity["name"].lower()
                    name = ''.join(c if c.isalnum() else '_' for c in name)
                    entity["id"] = f"{entity.get('type', 'entity')}_{name}_{i}"
            
            # Verify source and target exist
            valid_relations = []
            entity_ids = {entity.get("id") for entity in entities}
            
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                
                if source in entity_ids and target in entity_ids:
                    valid_relations.append(relation)
            
            return entities, valid_relations
        else:
            print("No JSON object found in the response")
            return [], []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing entity extraction response: {e}")
        print(f"Response: {result_text[:100]}...")  # Print just the first 100 chars
        return [], []

def extract_query_entities(llm, query: str) -> List[str]:
    """
    Extract entities from a query using the LLM
    
    Args:
        llm: LLM model
        query: Query to extract entities from
        
    Returns:
        List of entity names
    """
    prompt = f"""Extract the key entities (topics, people, companies, places, etc.) from this query.
Return ONLY a valid JSON array of entity names, like this: ["Entity1", "Entity2"]
Do not include any explanations, just the JSON array.

Query: {query}

JSON array:"""
    
    response = llm.invoke(prompt)
    
    try:
        # Try to find JSON array in response
        if "[" in response and "]" in response:
            json_str = response[response.find("["):response.rfind("]")+1]
            try:
                entities = json.loads(json_str)
                if isinstance(entities, list):
                    return [str(e) for e in entities if e]  # Convert all to strings and filter empty
                else:
                    return [str(entities)]  # Handle case where a single entity is returned
            except json.JSONDecodeError:
                # If JSON parsing fails, try direct string extraction
                # Remove brackets and split by commas
                clean_json = json_str.strip("[]")
                # Handle both quoted and unquoted strings
                if '"' in clean_json or "'" in clean_json:
                    # Try to handle quoted strings with regex
                    import re
                    matches = re.findall(r'["\'](.*?)["\']', clean_json)
                    if matches:
                        return [m.strip() for m in matches if m.strip()]
                
                # Fallback to simple splitting
                return [e.strip().strip('"\'') for e in clean_json.split(",") if e.strip()]
        else:
            # Fallback for no brackets: treat as comma-separated list
            return [e.strip().strip('"\'') for e in response.split(",") if e.strip()]
    except Exception as e:
        print(f"Error extracting entities from query: {e}")
        # Last resort: extract any words that might be entities
        words = query.split()
        return [w for w in words if len(w) > 3 and w[0].isupper()]

def list_available_llm_models():
    """
    List all available LLM models
    
    Returns:
        Dictionary of available models
    """
    return AVAILABLE_LLM_MODELS
