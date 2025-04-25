"""
Knowledge graph module for GraphRAG system
"""
import os
import json
import pickle
import networkx as nx
import re
from typing import List, Dict, Any, Tuple, Set
from langchain_core.documents import Document
import matplotlib.pyplot as plt

def contains_thai(text: str) -> bool:
    """
    Check if text contains a significant amount of Thai characters
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains significant Thai content
    """
    # Thai Unicode range (approximate)
    thai_pattern = re.compile('[\u0E00-\u0E7F]')
    
    # Count Thai characters
    thai_chars = len(thai_pattern.findall(text))
    
    # Calculate percentage of Thai characters
    if len(text) > 0:
        thai_percentage = thai_chars / len(text)
        # Consider text as Thai if it contains more than 10% Thai characters
        return thai_percentage > 0.1
    
    return False

class KnowledgeGraph:
    """
    Knowledge graph for storing entities and relationships
    """
    def __init__(self, persist_directory: str):
        """
        Initialize knowledge graph
        
        Args:
            persist_directory: Directory to persist the graph
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.graph_path = os.path.join(persist_directory, "knowledge_graph.pkl")
        self.graph = nx.DiGraph()
        self.entity_map = {}  # Maps entity names to IDs
        
        # Load graph if it exists
        self.load_graph()
    
    def load_graph(self):
        """Load graph from disk if it exists"""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data.get('graph', nx.DiGraph())
                    self.entity_map = data.get('entity_map', {})
                print(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                print(f"Error loading knowledge graph: {e}")
                self.graph = nx.DiGraph()
                self.entity_map = {}
    
    def save_graph(self):
        """Save graph to disk"""
        try:
            with open(self.graph_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'entity_map': self.entity_map
                }, f)
            print(f"Saved knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
    
    def add_entity(self, entity_id: str, entity_name: str, entity_type: str, attributes: Dict[str, Any] = None):
        """
        Add entity to graph
        
        Args:
            entity_id: Unique identifier for the entity
            entity_name: Name of the entity
            entity_type: Type of the entity (person, organization, etc.)
            attributes: Additional attributes for the entity
        """
        if attributes is None:
            attributes = {}
        
        # Add node to graph
        self.graph.add_node(
            entity_id,
            name=entity_name,
            entity_type=entity_type,
            **attributes
        )
        
        # Update entity map
        self.entity_map[entity_name.lower()] = entity_id
    
    def add_relationship(self, source_id: str, target_id: str, relation_type: str, attributes: Dict[str, Any] = None):
        """
        Add relationship to graph
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of the relationship
            attributes: Additional attributes for the relationship
        """
        if attributes is None:
            attributes = {}
        
        # Check if nodes exist
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            print(f"Warning: Cannot add relationship {source_id} --[{relation_type}]--> {target_id} - nodes don't exist")
            return
        
        # Add edge to graph
        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            **attributes
        )
    
    def get_entity_by_name(self, name: str) -> str:
        """
        Get entity ID by name
        
        Args:
            name: Name of the entity
            
        Returns:
            Entity ID or None if not found
        """
        return self.entity_map.get(name.lower())
    
    def search_entities(self, query: str, limit: int = 5) -> List[str]:
        """
        Search for entities by name
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of entity IDs
        """
        query_lower = query.lower()
        results = []
        
        # Exact match
        if query_lower in self.entity_map:
            results.append(self.entity_map[query_lower])
        
        # Partial match
        if len(results) < limit:
            for name, entity_id in self.entity_map.items():
                if query_lower in name and entity_id not in results:
                    results.append(entity_id)
                    if len(results) >= limit:
                        break
        
        return results
    
    def get_neighbors(self, entity_id: str, max_hops: int = 1, relation_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get neighbors of entity
        
        Args:
            entity_id: ID of the entity
            max_hops: Maximum number of hops
            relation_types: Filter by relation types
            
        Returns:
            List of nodes and edges in the neighborhood
        """
        if entity_id not in self.graph.nodes:
            return []
        
        # Extract subgraph
        if max_hops == 1:
            # Direct neighbors only
            neighbors = set(self.graph.successors(entity_id)) | set(self.graph.predecessors(entity_id))
            subgraph = self.graph.subgraph([entity_id] + list(neighbors))
        else:
            # BFS to get n-hop neighborhood
            nodes = {entity_id}
            current_nodes = {entity_id}
            
            for _ in range(max_hops):
                next_nodes = set()
                for node in current_nodes:
                    next_nodes.update(self.graph.successors(node))
                    next_nodes.update(self.graph.predecessors(node))
                nodes.update(next_nodes)
                current_nodes = next_nodes
            
            subgraph = self.graph.subgraph(nodes)
        
        # Convert to list of dicts
        result = []
        
        # Add entities
        for node_id in subgraph.nodes:
            node_data = subgraph.nodes[node_id]
            result.append({
                "type": "entity",
                "id": node_id,
                "name": node_data.get("name", node_id),
                "entity_type": node_data.get("entity_type", "unknown"),
                "attributes": {k: v for k, v in node_data.items() if k not in ["name", "entity_type"]}
            })
        
        # Add relationships
        for source, target, edge_data in subgraph.edges(data=True):
            relation_type = edge_data.get("relation_type", "related_to")
            
            # Skip if filtering by relation type
            if relation_types and relation_type not in relation_types:
                continue
            
            result.append({
                "type": "relation",
                "source": source,
                "target": target,
                "source_name": subgraph.nodes[source].get("name", source),
                "target_name": subgraph.nodes[target].get("name", target),
                "relation_type": relation_type,
                "attributes": {k: v for k, v in edge_data.items() if k != "relation_type"}
            })
        
        return result
    
    def get_shortest_path(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """
        Get shortest path between entities
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            
        Returns:
            List of nodes and edges in the path
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return []
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source_id, target_id)
            
            # Extract subgraph
            subgraph = self.graph.subgraph(path)
            
            # Convert to list of dicts
            result = []
            
            # Add entities
            for node_id in subgraph.nodes:
                node_data = subgraph.nodes[node_id]
                result.append({
                    "type": "entity",
                    "id": node_id,
                    "name": node_data.get("name", node_id),
                    "entity_type": node_data.get("entity_type", "unknown"),
                    "attributes": {k: v for k, v in node_data.items() if k not in ["name", "entity_type"]}
                })
            
            # Add relationships
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i+1]
                edge_data = subgraph.edges[source, target]
                
                result.append({
                    "type": "relation",
                    "source": source,
                    "target": target,
                    "source_name": subgraph.nodes[source].get("name", source),
                    "target_name": subgraph.nodes[target].get("name", target),
                    "relation_type": edge_data.get("relation_type", "related_to"),
                    "attributes": {k: v for k, v in edge_data.items() if k != "relation_type"}
                })
            
            return result
        except nx.NetworkXNoPath:
            return []
    
    def extract_and_add_from_documents(self, documents: List[Document], llm=None):
        """
        Extract entities and relationships from documents and add to graph
        
        Args:
            documents: List of documents
            llm: LLM model (if None, a new one will be created)
        """
        from rag.llm import get_llm_model, extract_entities_relations
        from rag.llm.thai_entity_extraction import extract_thai_entities_relations
        
        if llm is None:
            llm = get_llm_model()
        
        print(f"Extracting entities and relationships from {len(documents)} documents...")
        
        total_documents = len(documents)
        successful_documents = 0
        total_entities = 0
        total_relations = 0
        
        # Process documents in smaller batches for better entity extraction
        batch_size = min(5, total_documents)  # Process max 5 documents at once
        
        for i in range(0, total_documents, batch_size):
            batch_end = min(i + batch_size, total_documents)
            batch = documents[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(total_documents + batch_size - 1)//batch_size}...")
            
            for j, doc in enumerate(batch):
                doc_index = i + j + 1
                print(f"Processing document {doc_index}/{total_documents}...")
                
                try:
                    # Process only a portion of very large documents
                    content = doc.page_content
                    if len(content) > 4000:
                        content = content[:4000]  # Use first 4000 chars for large documents
                
                    # Check if content is primarily Thai
                    is_thai = contains_thai(content)
                    
                    # Use Thai-specific extraction for Thai content
                    if is_thai:
                        print(f"  Detected Thai content, using Thai-specific extraction")
                        entities, relations = extract_thai_entities_relations(llm, content)
                    else:
                        entities, relations = extract_entities_relations(llm, content)
                    
                    if not entities:
                        print(f"  No entities found in document {doc_index}")
                        continue
                    
                    # Add source metadata
                    source = doc.metadata.get('source', 'Unknown')
                    
                    # Add entities to graph
                    for entity in entities:
                        entity_id = entity.get('id')
                        entity_name = entity.get('name')
                        entity_type = entity.get('type')
                        
                        if not all([entity_id, entity_name, entity_type]):
                            continue
                        
                        # Add source to attributes
                        attributes = entity.get('attributes', {})
                        attributes['sources'] = attributes.get('sources', []) + [source]
                        
                        # Add entity to graph
                        self.add_entity(entity_id, entity_name, entity_type, attributes)
                        total_entities += 1
                    
                    # Add relationships to graph
                    for relation in relations:
                        source_id = relation.get('source')
                        target_id = relation.get('target')
                        relation_type = relation.get('type')
                        
                        if not all([source_id, target_id, relation_type]):
                            continue
                        
                        # Add source to attributes
                        attributes = relation.get('attributes', {})
                        attributes['sources'] = attributes.get('sources', []) + [source]
                        
                        # Add relationship to graph
                        self.add_relationship(source_id, target_id, relation_type, attributes)
                        total_relations += 1
                    
                    successful_documents += 1
                    print(f"  Added {len(entities)} entities and {len(relations)} relations from document {doc_index}")
                
                except Exception as e:
                    print(f"Error processing document {doc_index}: {e}")
            
            # Save graph after each batch
            if self.graph.number_of_nodes() > 0:
                self.save_graph()
                print(f"Intermediate save: {self.graph.number_of_nodes()} entities and {self.graph.number_of_edges()} relationships")
        
        # Final save
        self.save_graph()
        
        print(f"Processing complete: {successful_documents}/{total_documents} documents processed successfully")
        print(f"Knowledge graph now has {self.graph.number_of_nodes()} entities and {self.graph.number_of_edges()} relationships")
    
    def visualize(self, output_path: str = None, max_nodes: int = 50):
        """
        Visualize the knowledge graph
        
        Args:
            output_path: Path to save the visualization
            max_nodes: Maximum number of nodes to display
        """
        if self.graph.number_of_nodes() == 0:
            print("Knowledge graph is empty, nothing to visualize.")
            return
        
        # Limit the number of nodes for visualization
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Graph is too large ({self.graph.number_of_nodes()} nodes), showing top {max_nodes} nodes by degree")
            
            # Get nodes sorted by degree
            nodes_by_degree = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes = [node for node, _ in nodes_by_degree]
            
            # Create subgraph
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        # Setup Thai font support
        from rag.visualization.visualization import setup_thai_font
        setup_thai_font()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create node colors based on entity type
        entity_types = [subgraph.nodes[node].get('entity_type', 'unknown') for node in subgraph.nodes]
        unique_types = list(set(entity_types))
        color_map = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
        node_colors = [color_map[t] for t in entity_types]
        
        # Create node labels
        node_labels = {node: subgraph.nodes[node].get('name', node) for node in subgraph.nodes}
        
        # Create edge labels
        edge_labels = {(u, v): d.get('relation_type', '') for u, v, d in subgraph.edges(data=True)}
        
        # Create layout
        pos = nx.spring_layout(subgraph, seed=42, k=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
        
        # Draw node labels
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=10)
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     label=entity_type, markersize=10) 
                          for entity_type, color in color_map.items()]
        plt.legend(handles=legend_elements, title="Entity Types", loc='best')
        
        # Set title
        plt.title(f"Knowledge Graph ({subgraph.number_of_nodes()} entities, {subgraph.number_of_edges()} relationships)")
        
        # Remove axis
        plt.axis('off')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    def to_mermaid(self, output_path: str, max_nodes: int = 50):
        """
        Generate a Mermaid diagram representation of the knowledge graph
        
        Args:
            output_path: Path to save the Mermaid diagram
            max_nodes: Maximum number of nodes to display
        """
        if self.graph.number_of_nodes() == 0:
            print("Knowledge graph is empty, nothing to visualize.")
            return
        
        # Limit the number of nodes for visualization
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Graph is too large ({self.graph.number_of_nodes()} nodes), showing top {max_nodes} nodes by degree")
            
            # Get nodes sorted by degree
            nodes_by_degree = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes = [node for node, _ in nodes_by_degree]
            
            # Create subgraph
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        # Create Mermaid content
        mermaid_content = ["```mermaid", "graph TD"]
        
        # Add style classes for different entity types
        entity_types = set(subgraph.nodes[node].get('entity_type', 'unknown') for node in subgraph.nodes)
        style_defs = []
        for i, entity_type in enumerate(entity_types):
            color = f"#{(i*3+1)%10}{(i*5+3)%10}{(i*7+5)%10}"
            style_defs.append(f"    classDef {entity_type} fill:{color},stroke:#333,stroke-width:1px")
        
        # Add nodes
        for node in subgraph.nodes:
            node_name = subgraph.nodes[node].get('name', node)
            node_name = node_name.replace('"', '\\"')  # Escape quotes
            mermaid_content.append(f"    {node}[\"{node_name}\"]")
            entity_type = subgraph.nodes[node].get('entity_type', 'unknown')
            mermaid_content.append(f"    class {node} {entity_type}")
        
        # Add edges
        for u, v, data in subgraph.edges(data=True):
            relation_type = data.get('relation_type', '')
            relation_type = relation_type.replace('"', '\\"')  # Escape quotes
            mermaid_content.append(f"    {u} -->|{relation_type}| {v}")
        
        # Add style definitions
        mermaid_content.extend(style_defs)
        mermaid_content.append("```")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(mermaid_content))
        
        print(f"Mermaid diagram saved to {output_path}")
