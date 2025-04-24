"""
Visualization module for GraphRAG
"""
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any

def visualize_graph(knowledge_graph, output_path: str = None, max_nodes: int = 50):
    """
    Visualize a knowledge graph
    
    Args:
        knowledge_graph: KnowledgeGraph instance
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to display
    """
    knowledge_graph.visualize(output_path, max_nodes)

def visualize_query_path(knowledge_graph, graph_data: List[Dict[str, Any]], output_path: str):
    """
    Visualize the query path
    
    Args:
        knowledge_graph: KnowledgeGraph instance
        graph_data: Graph data from retrieval
        output_path: Path to save the visualization
    """
    # Create a new directed graph for visualization
    G = nx.DiGraph()
    
    # Extract entities and relations
    entities = {}
    relations = []
    
    for item in graph_data:
        if item["type"] == "entity":
            entity_id = item["id"]
            entities[entity_id] = item
            
            # Add node
            G.add_node(
                entity_id,
                name=item["name"],
                entity_type=item["entity_type"]
            )
        elif item["type"] == "relation":
            relations.append(item)
            
            # Add edge
            G.add_edge(
                item["source"],
                item["target"],
                relation_type=item["relation_type"]
            )
    
    if len(G.nodes) == 0:
        print("No entities found in graph data.")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create node colors based on entity type
    entity_types = [G.nodes[node].get('entity_type', 'unknown') for node in G.nodes]
    unique_types = list(set(entity_types))
    color_map = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
    node_colors = [color_map[t] for t in entity_types]
    
    # Create node labels
    node_labels = {node: G.nodes[node].get('name', node) for node in G.nodes}
    
    # Create edge labels
    edge_labels = {(u, v): d.get('relation_type', '') for u, v, d in G.edges(data=True)}
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  label=entity_type, markersize=10) 
                      for entity_type, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Entity Types", loc='best')
    
    # Set title
    plt.title(f"Query Path ({len(G.nodes)} entities, {len(G.edges)} relationships)")
    
    # Remove axis
    plt.axis('off')
    
    # Save
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()
