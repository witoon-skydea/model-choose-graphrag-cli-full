"""
Retrieval module for GraphRAG system
"""
from typing import List, Dict, Any
import json
from langchain_core.documents import Document
from rag.vector_store import similarity_search
from rag.knowledge_graph.graph import KnowledgeGraph
from rag.llm.llm import get_llm_model, extract_query_entities

def merge_and_rank_results(vector_results: List[Document], graph_results: List[Document], query: str) -> List[Document]:
    """
    Merge and rank results from vector search and graph search
    
    Args:
        vector_results: Results from vector search
        graph_results: Results from graph search
        query: Original query
        
    Returns:
        Merged and ranked results
    """
    # Combine results
    combined_results = list(vector_results)
    
    # Add graph results that are not already in vector results
    vector_contents = {doc.page_content for doc in vector_results}
    for doc in graph_results:
        if doc.page_content not in vector_contents:
            combined_results.append(doc)
    
    return combined_results

def convert_graph_to_documents(graph_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert graph data to documents
    
    Args:
        graph_data: Graph data from KnowledgeGraph
        
    Returns:
        List of documents
    """
    documents = []
    
    # Group by entities
    entities = {}
    relations = []
    
    for item in graph_data:
        if item["type"] == "entity":
            entity_id = item["id"]
            entities[entity_id] = item
        elif item["type"] == "relation":
            relations.append(item)
    
    # Create documents for entities with their relations
    for entity_id, entity in entities.items():
        # Find relations for this entity
        entity_relations = [r for r in relations if r["source"] == entity_id or r["target"] == entity_id]
        
        # Create content
        content = f"Entity: {entity['name']} (Type: {entity['entity_type']})\n"
        
        # Add attributes
        attributes = entity.get("attributes", {})
        if attributes:
            content += "Attributes:\n"
            for key, value in attributes.items():
                if key != "sources":
                    content += f"- {key}: {value}\n"
        
        # Add relations
        if entity_relations:
            content += "Relationships:\n"
            for rel in entity_relations:
                if rel["source"] == entity_id:
                    content += f"- {rel['relation_type']} -> {rel['target_name']}\n"
                else:
                    content += f"- {rel['source_name']} -> {rel['relation_type']} -> {entity['name']}\n"
        
        # Create document
        sources = attributes.get("sources", ["knowledge_graph"])
        doc = Document(
            page_content=content,
            metadata={
                "source": "knowledge_graph",
                "entity_id": entity_id,
                "entity_name": entity["name"],
                "entity_type": entity["entity_type"],
                "original_sources": sources
            }
        )
        documents.append(doc)
    
    return documents

def hybrid_retrieval(vector_store, knowledge_graph: KnowledgeGraph, query: str, llm=None, k: int = 4, max_hops: int = 1) -> List[Document]:
    """
    Perform hybrid retrieval using both vector store and knowledge graph
    
    Args:
        vector_store: Vector store
        knowledge_graph: Knowledge graph
        query: Query to search for
        llm: LLM model (if None, a new one will be created)
        k: Number of results to return
        max_hops: Maximum number of hops for graph traversal
        
    Returns:
        List of documents
    """
    if llm is None:
        llm = get_llm_model()
    
    # Step 1: Vector search to find relevant documents
    print("Performing vector search...")
    vector_results = similarity_search(vector_store, query, k=k)
    
    # Step 2: Extract entities from the query
    print("Extracting entities from query...")
    entities = extract_query_entities(llm, query)
    print(f"Extracted entities: {entities}")
    
    # Step 3: Find related entities in the graph
    print("Searching knowledge graph...")
    graph_results = []
    
    for entity in entities:
        # Search for entity in graph
        entity_ids = knowledge_graph.search_entities(entity, limit=2)
        
        for entity_id in entity_ids:
            # Get neighbors up to max_hops away
            neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=max_hops)
            
            # Convert graph results to documents
            if neighbors:
                graph_docs = convert_graph_to_documents(neighbors)
                graph_results.extend(graph_docs)
    
    print(f"Found {len(graph_results)} relevant items in knowledge graph")
    
    # Step 4: Combine and rank results
    combined_results = merge_and_rank_results(vector_results, graph_results, query)
    
    return combined_results[:k*2]  # Return more results since we have two sources
