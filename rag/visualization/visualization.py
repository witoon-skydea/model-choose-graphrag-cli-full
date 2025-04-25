"""
Visualization module for GraphRAG
"""
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any
import os
import matplotlib.font_manager as fm
from pathlib import Path

def visualize_graph(knowledge_graph, output_path: str = None, max_nodes: int = 50, format: str = "png"):
    """
    Visualize a knowledge graph
    
    Args:
        knowledge_graph: KnowledgeGraph instance
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to display
        format: Output format ('png' or 'mermaid')
    """
    if format.lower() == "mermaid":
        # Extract mermaid filename from output_path
        if output_path and output_path.lower().endswith('.png'):
            output_path = output_path[:-4] + '.md'
        elif not output_path:
            output_path = os.path.join(os.path.dirname(knowledge_graph.persist_directory), "knowledge_graph.md")
        
        # Generate Mermaid diagram
        knowledge_graph.to_mermaid(output_path, max_nodes)
    else:
        # Regular PNG visualization
        knowledge_graph.visualize(output_path, max_nodes)

def visualize_query_path(knowledge_graph, graph_data: List[Dict[str, Any]], output_path: str, format: str = "png"):
    """
    Visualize the query path
    
    Args:
        knowledge_graph: KnowledgeGraph instance
        graph_data: Graph data from retrieval
        output_path: Path to save the visualization
        format: Output format ('png' or 'mermaid')
    """
    if format.lower() == "mermaid":
        # Extract mermaid filename from output_path
        if output_path and output_path.lower().endswith('.png'):
            output_path = output_path[:-4] + '.md'
        
        # Create Mermaid diagram
        create_query_path_mermaid(graph_data, output_path)
        return
    
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
    
    # Set up Thai font support if needed
    setup_thai_font()
    
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

def setup_thai_font():
    """
    Setup Thai font for matplotlib
    """
    # Try to find a font that supports Thai characters
    # Common Thai fonts: TH Sarabun New, Angsana New, Garuda, Norasi, Waree
    thai_fonts = ['TH Sarabun New', 'Angsana New', 'Garuda', 'Norasi', 'Waree', 
                 'Tahoma', 'Arial Unicode MS', 'DejaVu Sans']
    
    # Check if any of the Thai fonts is available
    font_found = False
    for font_name in thai_fonts:
        font_list = [f.name for f in fm.fontManager.ttflist]
        if font_name in font_list:
            plt.rcParams['font.family'] = font_name
            print(f"Using font '{font_name}' for Thai text support")
            font_found = True
            break
    
    if not font_found:
        print("Warning: No Thai font found in the system. Thai characters may not display correctly.")
        print("Consider installing a Thai font like 'TH Sarabun New' or 'Garuda'.")

def create_query_path_mermaid(graph_data: List[Dict[str, Any]], output_path: str):
    """
    Create a Mermaid diagram of the query path
    
    Args:
        graph_data: Graph data from retrieval
        output_path: Path to save the Mermaid diagram
    """
    # Extract entities and relations
    entities = {}
    relations = []
    
    for item in graph_data:
        if item["type"] == "entity":
            entity_id = item["id"]
            entities[entity_id] = item
        elif item["type"] == "relation":
            relations.append(item)
    
    if not entities:
        print("No entities found in graph data.")
        return
    
    # Create Mermaid content
    mermaid_content = ["```mermaid", "graph TD"]
    
    # Add style classes for different entity types
    entity_types = set(item["entity_type"] for item in entities.values())
    style_defs = []
    for i, entity_type in enumerate(entity_types):
        color = f"#{(i*3+1)%10}{(i*5+3)%10}{(i*7+5)%10}"
        style_defs.append(f"    classDef {entity_type} fill:{color},stroke:#333,stroke-width:1px")
    
    # Add nodes
    for entity_id, entity in entities.items():
        entity_name = entity["name"].replace('"', '\\"')  # Escape quotes
        mermaid_content.append(f"    {entity_id}[\"{entity_name}\"]")
        mermaid_content.append(f"    class {entity_id} {entity['entity_type']}")
    
    # Add relationships
    for relation in relations:
        source_id = relation["source"]
        target_id = relation["target"]
        relation_type = relation["relation_type"].replace('"', '\\"')  # Escape quotes
        mermaid_content.append(f"    {source_id} -->|{relation_type}| {target_id}")
    
    # Add style definitions
    mermaid_content.extend(style_defs)
    mermaid_content.append("```")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(mermaid_content))
    
    print(f"Mermaid diagram saved to {output_path}")
