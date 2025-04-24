#!/usr/bin/env python3
"""
Model-Choose GraphRAG CLI - Enhanced RAG CLI with knowledge graph and model selection capabilities
"""
import os
import argparse
import sys
from rag.document_loader import load_document, scan_directory, is_supported_file
from rag.vector_store import (
    get_vector_store, add_documents, similarity_search,
    list_companies, get_active_company, add_company, remove_company, set_active_company,
    set_company_models, get_system_settings, set_system_settings
)
from rag.llm import get_llm_model, generate_response, list_available_llm_models
from rag.embeddings import get_embeddings_model, list_available_embedding_models
from rag.knowledge_graph import KnowledgeGraph
from rag.retrieval import hybrid_retrieval
from rag.visualization import visualize_graph, visualize_query_path
from rag.config import CompanyConfig, SystemConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_GRAPH_DIR = "graph"

def ingest_documents(args):
    """
    Ingest documents into the vector store and optionally the knowledge graph
    
    Args:
        args: Command line arguments
    """
    # Get vector store for the specified company
    config = CompanyConfig()
    
    if args.company:
        try:
            db_path = config.get_db_path(args.company)
            company_models = config.get_company_model_settings(args.company)
            embedding_model = args.embedding_model or company_models.get("embedding_model")
            vector_store = get_vector_store(db_path, embedding_model=embedding_model)
            active_company = args.company
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available companies: {', '.join(config.get_companies())}")
            sys.exit(1)
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
        embedding_model = args.embedding_model or company_models.get("embedding_model")
        vector_store = get_vector_store(db_path, embedding_model=embedding_model)
    
    # Show active company and model info
    company_details = config.get_company_details(active_company)
    print(f"Active company: {company_details.get('name')} ({active_company})")
    print(f"Vector store location: {db_path}")
    print(f"Using embedding model: {embedding_model}")
    
    # Get knowledge graph if building graph is requested
    knowledge_graph = None
    if args.build_graph:
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        knowledge_graph = KnowledgeGraph(graph_dir)
        print(f"Knowledge graph location: {graph_dir}")
    
    # Prepare list of files to ingest
    files_to_ingest = []
    
    # Process each input path (file or directory)
    for path in args.paths:
        if os.path.isdir(path):
            # If it's a directory, scan for supported files
            print(f"Scanning directory: {path}...")
            try:
                dir_files = scan_directory(path, recursive=args.recursive)
                print(f"  Found {len(dir_files)} supported files")
                files_to_ingest.extend(dir_files)
            except Exception as e:
                print(f"Error scanning directory {path}: {e}")
        elif os.path.isfile(path):
            # If it's a file, check if it's supported
            if is_supported_file(path):
                files_to_ingest.append(path)
            else:
                print(f"Skipping unsupported file: {path}")
        else:
            print(f"Path not found: {path}")
    
    # Load and add documents
    total_files = len(files_to_ingest)
    successful_files = 0
    all_documents = []  # Store all documents for knowledge graph building
    
    # Set up OCR options if OCR is enabled
    ocr_options = None
    if args.ocr:
        ocr_options = {
            'engine': args.ocr_engine,
            'lang': args.ocr_lang,
            'dpi': args.ocr_dpi,
            'use_gpu': args.gpu,
            'tesseract_cmd': args.tesseract_cmd,
            'tessdata_dir': args.tessdata_dir
        }
    
    # Get chunk settings from system config
    sys_config = SystemConfig()
    chunk_settings = sys_config.get_chunk_settings()
    chunk_size = args.chunk_size or chunk_settings.get("chunk_size")
    chunk_overlap = args.chunk_overlap or chunk_settings.get("chunk_overlap")
    
    for i, file_path in enumerate(files_to_ingest, 1):
        print(f"[{i}/{total_files}] Loading {file_path}...")
        try:
            documents = load_document(
                file_path, 
                ocr_enabled=args.ocr, 
                ocr_options=ocr_options,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            add_documents(vector_store, documents)
            all_documents.extend(documents)  # Store for knowledge graph
            print(f"  Loaded and added {len(documents)} chunks")
            successful_files += 1
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"\nIngestion complete: {successful_files}/{total_files} files processed successfully")
    print(f"Vector store location: {db_path}")
    print(f"Used chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
    
    # Build knowledge graph if requested
    if args.build_graph and all_documents:
        print("\nBuilding knowledge graph from documents...")
        llm = get_llm_model(args.llm_model or company_models.get("llm_model"))
        knowledge_graph.extract_and_add_from_documents(all_documents, llm)
        
        # Visualize graph if requested
        if args.visualize_graph:
            graph_viz_path = os.path.join(os.path.dirname(db_path), "knowledge_graph.png")
            print(f"Visualizing knowledge graph to {graph_viz_path}...")
            knowledge_graph.visualize(graph_viz_path, max_nodes=50)

def answer_question(args):
    """
    Answer a question using the GraphRAG system
    
    Args:
        args: Command line arguments
    """
    # Get vector store and LLM for the specified company
    config = CompanyConfig()
    sys_config = SystemConfig()
    
    if args.company:
        try:
            db_path = config.get_db_path(args.company)
            company_models = config.get_company_model_settings(args.company)
            active_company = args.company
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available companies: {', '.join(config.get_companies())}")
            sys.exit(1)
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
    
    if not os.path.exists(db_path):
        print(f"Error: Vector store not found at {db_path}")
        print("Please ingest documents first using the 'ingest' command")
        sys.exit(1)
    
    # Get embedding model (default or override)
    embedding_model = args.embedding_model or company_models.get("embedding_model")
    
    # Show active company
    company_details = config.get_company_details(active_company)
    print(f"Active company: {company_details.get('name')} ({active_company})")
    
    # Get vector store
    vector_store = get_vector_store(db_path, embedding_model=embedding_model)
    
    # Get top_k from arguments or system settings
    top_k = args.num_chunks or sys_config.get_top_k()
    
    # Get knowledge graph if using graph or hybrid retrieval
    knowledge_graph = None
    graph_data = None
    if args.retrieval_method in ["graph", "hybrid"]:
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
            print(f"Warning: Knowledge graph not found at {graph_dir}")
            print("Falling back to vector search only")
            args.retrieval_method = "vector"
        else:
            knowledge_graph = KnowledgeGraph(graph_dir)
            print(f"Using knowledge graph from: {graph_dir}")
    
    # Get documents based on retrieval method
    if args.retrieval_method == "vector" or knowledge_graph is None:
        # Use vector search only
        print(f"Searching for relevant documents using embedding model: {embedding_model}...")
        documents = similarity_search(vector_store, args.question, k=top_k)
    elif args.retrieval_method == "graph":
        # Use graph search only
        print("Using graph search only...")
        # Get LLM for entity extraction
        llm_model = args.llm_model or company_models.get("llm_model")
        temperature = args.temperature or sys_config.get_temperature()
        llm = get_llm_model(llm_model, temperature)
        
        from rag.llm.llm import extract_query_entities
        entities = extract_query_entities(llm, args.question)
        
        graph_data = []
        for entity in entities:
            entity_ids = knowledge_graph.search_entities(entity, limit=2)
            for entity_id in entity_ids:
                neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=args.num_hops)
                graph_data.extend(neighbors)
        
        # Convert to documents format
        from rag.retrieval import convert_graph_to_documents
        documents = convert_graph_to_documents(graph_data)
    else:
        # Use hybrid search (default)
        print("Using hybrid search (vector + knowledge graph)...")
        # Get LLM for hybrid search
        llm_model = args.llm_model or company_models.get("llm_model")
        temperature = args.temperature or sys_config.get_temperature()
        llm = get_llm_model(llm_model, temperature)
        
        documents = hybrid_retrieval(
            vector_store, 
            knowledge_graph, 
            args.question, 
            llm,
            k=top_k,
            max_hops=args.num_hops
        )
        
        # Extract graph data for explanation if needed
        if args.explain:
            graph_data = []
            for doc in documents:
                if doc.metadata.get('source') == 'knowledge_graph':
                    entity_id = doc.metadata.get('entity_id')
                    if entity_id:
                        neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                        for item in neighbors:
                            if item not in graph_data:
                                graph_data.append(item)
    
    if args.raw_chunks:
        # Print raw chunks without LLM processing
        print("-" * 80)
        print(f"Top {len(documents)} relevant chunks:")
        for i, doc in enumerate(documents):
            print(f"\nChunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
            print("-" * 40)
            print(doc.page_content)
        print("-" * 80)
    else:
        # Get LLM model (default or override)
        llm_model = args.llm_model or company_models.get("llm_model")
        temperature = args.temperature or sys_config.get_temperature()
        
        print(f"Using LLM model: {llm_model} (temperature: {temperature})")
        llm = get_llm_model(llm_model, temperature)
        
        # Generate response
        print("Generating response...\n")
        
        # Use custom template if provided
        custom_template = None
        if args.template:
            try:
                with open(args.template, 'r', encoding='utf-8') as f:
                    custom_template = f.read()
                print(f"Using custom prompt template from {args.template}")
            except Exception as e:
                print(f"Error loading template from {args.template}: {e}")
                print("Using default template instead.")
        
        # Generate response with graph data if available
        response = generate_response(llm, documents, args.question, custom_template, graph_data)
        
        # Print response
        print("-" * 80)
        print("Answer:")
        print(response.strip())
        print("-" * 80)
    
    # Visualize graph used in query if requested
    if args.explain and graph_data:
        output_path = os.path.join(os.path.dirname(db_path), "query_explanation.png")
        print(f"Generating visual explanation to {output_path}...")
        visualize_query_path(knowledge_graph, graph_data, output_path)

def build_graph(args):
    """
    Build knowledge graph from existing vector store
    
    Args:
        args: Command line arguments
    """
    # Get vector store and LLM for the specified company
    config = CompanyConfig()
    
    if args.company:
        try:
            db_path = config.get_db_path(args.company)
            company_models = config.get_company_model_settings(args.company)
            active_company = args.company
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available companies: {', '.join(config.get_companies())}")
            sys.exit(1)
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
    
    if not os.path.exists(db_path):
        print(f"Error: Vector store not found at {db_path}")
        print("Please ingest documents first using the 'ingest' command")
        sys.exit(1)
    
    # Get vector store
    embedding_model = args.embedding_model or company_models.get("embedding_model")
    vector_store = get_vector_store(db_path, embedding_model=embedding_model)
    
    # Get knowledge graph
    graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
    knowledge_graph = KnowledgeGraph(graph_dir)
    
    # Get LLM model
    llm_model = args.llm_model or company_models.get("llm_model")
    llm = get_llm_model(llm_model)
    
    # Get documents from vector store
    print("Retrieving documents from vector store...")
    
    # Since Chroma doesn't have a direct method to get all documents,
    # we need to retrieve them first using a generic query
    if args.query:
        print(f"Using query: {args.query}")
        # Use the provided query to find relevant documents
        documents = similarity_search(vector_store, args.query, k=args.num_docs)
    else:
        # Use a generic query to retrieve documents
        print("No query provided, using a generic query to retrieve documents")
        documents = similarity_search(vector_store, "summarize all information", k=args.num_docs)
    
    print(f"Retrieved {len(documents)} documents from vector store")
    
    # Build knowledge graph
    print("Building knowledge graph from documents...")
    knowledge_graph.extract_and_add_from_documents(documents, llm)
    
    print(f"Knowledge graph location: {graph_dir}")
    
    # Visualize graph if requested
    if args.visualize:
        output_path = os.path.join(os.path.dirname(db_path), "knowledge_graph.png")
        print(f"Visualizing knowledge graph to {output_path}...")
        knowledge_graph.visualize(output_path, max_nodes=args.max_nodes)

def visualize_knowledge_graph(args):
    """
    Visualize the knowledge graph
    
    Args:
        args: Command line arguments
    """
    # Get company path
    config = CompanyConfig()
    
    if args.company:
        try:
            db_path = config.get_db_path(args.company)
            active_company = args.company
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available companies: {', '.join(config.get_companies())}")
            sys.exit(1)
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
    
    # Get knowledge graph
    graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
    if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
        print(f"Error: Knowledge graph not found at {graph_dir}")
        print("Please build the knowledge graph first using the 'build-graph' command")
        sys.exit(1)
    
    knowledge_graph = KnowledgeGraph(graph_dir)
    
    # Visualize graph
    output_path = args.output if args.output else os.path.join(os.path.dirname(db_path), "knowledge_graph.png")
    print(f"Visualizing knowledge graph to {output_path}...")
    knowledge_graph.visualize(output_path, max_nodes=args.max_nodes)

def company_management(args):
    """
    Manage companies in the system
    
    Args:
        args: Command line arguments
    """
    if args.list:
        # List all companies
        companies = list_companies()
        
        print("\nAvailable Companies:")
        print("-" * 80)
        print(f"{'ID':<15} {'NAME':<20} {'DESCRIPTION':<20} {'LLM MODEL':<15} {'EMBED MODEL':<15} {'ACTIVE'}")
        print("-" * 80)
        
        for company in companies:
            active_mark = "✓" if company.get('active', False) else ""
            print(f"{company['id']:<15} {company['name']:<20} {company['description'][:20]:<20} {company['llm_model']:<15} {company['embedding_model']:<15} {active_mark}")
        
        print("-" * 80)
        
    elif args.add:
        # Add a new company
        try:
            add_company(args.id, args.name, args.description or "", args.llm_model, args.embedding_model)
            print(f"Company '{args.name}' ({args.id}) added successfully")
            
            # Automatically set as active if requested
            if args.set_active:
                set_active_company(args.id)
                print(f"Company '{args.id}' set as active")
                
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.remove:
        # Remove a company
        try:
            remove_company(args.id)
            print(f"Company '{args.id}' removed successfully")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.set_active:
        # Set active company
        try:
            set_active_company(args.id)
            print(f"Company '{args.id}' set as active")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.show_active:
        # Show active company
        active_company = get_active_company()
        
        print("\nActive Company:")
        print("-" * 60)
        print(f"ID:              {active_company['id']}")
        print(f"Name:            {active_company['name']}")
        print(f"Description:     {active_company['description']}")
        print(f"DB Path:         {active_company['db_dir']}")
        print(f"LLM Model:       {active_company['llm_model']}")
        print(f"Embedding Model: {active_company['embedding_model']}")
        print("-" * 60)
            
    elif args.set_models:
        # Set models for a company
        try:
            set_company_models(args.id, args.llm_model, args.embedding_model)
            print(f"Models updated for company '{args.id}':")
            
            # Show updated model settings
            config = CompanyConfig()
            model_settings = config.get_company_model_settings(args.id)
            print(f"LLM Model:       {model_settings.get('llm_model')}")
            print(f"Embedding Model: {model_settings.get('embedding_model')}")
            
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("No action specified. Use --list, --add, --remove, --set-active, --show-active, or --set-models.")
        sys.exit(1)

def model_management(args):
    """
    Manage models and system settings
    
    Args:
        args: Command line arguments
    """
    if args.list_llm or args.list_all:
        # List all available LLM models
        llm_models = list_available_llm_models()
        
        print("\nAvailable LLM Models:")
        print("-" * 60)
        for key, value in llm_models.items():
            print(f"{key:<15} => {value}")
        print("-" * 60)
    
    if args.list_embeddings or args.list_all:
        # List all available embedding models
        embedding_models = list_available_embedding_models()
        
        print("\nAvailable Embedding Models:")
        print("-" * 60)
        for key, value in embedding_models.items():
            print(f"{key:<15} => {value}")
        print("-" * 60)
    
    if args.show_settings:
        # Show current system settings
        settings = get_system_settings()
        
        print("\nCurrent System Settings:")
        print("-" * 60)
        print(f"Default LLM Model:       {settings.get('default_llm_model')}")
        print(f"Default Embedding Model: {settings.get('default_embedding_model')}")
        print(f"Temperature:             {settings.get('temperature')}")
        print(f"Top K Results:           {settings.get('top_k')}")
        print(f"Chunk Size:              {settings.get('chunk_size')}")
        print(f"Chunk Overlap:           {settings.get('chunk_overlap')}")
        print("-" * 60)
    
    if args.set_defaults:
        # Update system settings
        settings_to_update = {}
        
        if args.default_llm:
            settings_to_update["default_llm_model"] = args.default_llm
            
        if args.default_embedding:
            settings_to_update["default_embedding_model"] = args.default_embedding
            
        if args.temperature is not None:
            settings_to_update["temperature"] = args.temperature
            
        if args.top_k is not None:
            settings_to_update["top_k"] = args.top_k
            
        if args.chunk_size is not None and args.chunk_overlap is not None:
            settings_to_update["chunk_size"] = args.chunk_size
            settings_to_update["chunk_overlap"] = args.chunk_overlap
        
        if settings_to_update:
            # Update settings
            set_system_settings(settings_to_update)
            print("System settings updated successfully.")
            
            # Show updated settings
            settings = get_system_settings()
            print("\nUpdated System Settings:")
            print("-" * 60)
            print(f"Default LLM Model:       {settings.get('default_llm_model')}")
            print(f"Default Embedding Model: {settings.get('default_embedding_model')}")
            print(f"Temperature:             {settings.get('temperature')}")
            print(f"Top K Results:           {settings.get('top_k')}")
            print(f"Chunk Size:              {settings.get('chunk_size')}")
            print(f"Chunk Overlap:           {settings.get('chunk_overlap')}")
            print("-" * 60)
        else:
            print("No settings specified to update.")
            sys.exit(1)

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="Model-Choose GraphRAG CLI - Enhanced PDF RAG CLI with knowledge graph and model selection capabilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Company management command
    company_parser = subparsers.add_parser("company", help="Manage companies")
    company_parser.add_argument("--list", action="store_true", help="List all companies")
    company_parser.add_argument("--add", action="store_true", help="Add a new company")
    company_parser.add_argument("--remove", action="store_true", help="Remove a company")
    company_parser.add_argument("--set-active", action="store_true", help="Set active company")
    company_parser.add_argument("--show-active", action="store_true", help="Show active company")
    company_parser.add_argument("--set-models", action="store_true", help="Set models for a company")
    company_parser.add_argument("--id", help="Company ID (required for add/remove/set-active/set-models)")
    company_parser.add_argument("--name", help="Company name (required for add)")
    company_parser.add_argument("--description", help="Company description (optional for add)")
    company_parser.add_argument("--llm-model", help="Custom LLM model for this company (optional for add/set-models)")
    company_parser.add_argument("--embedding-model", help="Custom embedding model for this company (optional for add/set-models)")
    
    # Model management command
    model_parser = subparsers.add_parser("model", help="Manage models and system settings")
    model_parser.add_argument("--list-llm", action="store_true", help="List available LLM models")
    model_parser.add_argument("--list-embeddings", action="store_true", help="List available embedding models")
    model_parser.add_argument("--list-all", action="store_true", help="List all available models")
    model_parser.add_argument("--show-settings", action="store_true", help="Show current system settings")
    model_parser.add_argument("--set-defaults", action="store_true", help="Set default system settings")
    model_parser.add_argument("--default-llm", help="Set default LLM model")
    model_parser.add_argument("--default-embedding", help="Set default embedding model")
    model_parser.add_argument("--temperature", type=float, help="Set default temperature")
    model_parser.add_argument("--top-k", type=int, help="Set default number of chunks to return")
    model_parser.add_argument("--chunk-size", type=int, help="Set default chunk size")
    model_parser.add_argument("--chunk-overlap", type=int, help="Set default chunk overlap")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    ingest_parser.add_argument("--company", help="Company ID to ingest documents for (uses active company if not specified)")
    ingest_parser.add_argument("--recursive", action="store_true", default=True, 
                              help="Recursively scan directories for files (default: True)")
    ingest_parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                              help="Do not recursively scan directories")
    ingest_parser.add_argument("--embedding-model", help="Override embedding model for this ingestion")
    ingest_parser.add_argument("--chunk-size", type=int, help="Override chunk size for this ingestion")
    ingest_parser.add_argument("--chunk-overlap", type=int, help="Override chunk overlap for this ingestion")
    ingest_parser.add_argument("--build-graph", action="store_true", default=False,
                              help="Build knowledge graph during ingestion")
    ingest_parser.add_argument("--llm-model", help="LLM model to use for entity extraction")
    ingest_parser.add_argument("--visualize-graph", action="store_true", default=False,
                              help="Visualize the knowledge graph after building")
    
    # OCR options
    ingest_parser.add_argument("--ocr", action="store_true", default=False,
                              help="Enable OCR for PDFs with images")
    ingest_parser.add_argument("--ocr-engine", choices=["tesseract", "easyocr"], default="tesseract",
                              help="OCR engine to use (default: tesseract)")
    ingest_parser.add_argument("--ocr-lang", default="eng",
                              help="Language code(s) for OCR (e.g., 'eng' for English, 'tha+eng' for Thai and English)")
    ingest_parser.add_argument("--ocr-dpi", type=int, default=300,
                              help="DPI setting for PDF to image conversion (default: 300)")
    ingest_parser.add_argument("--tesseract-cmd", 
                              help="Path to Tesseract executable (if not in PATH)")
    ingest_parser.add_argument("--tessdata-dir",
                              help="Path to directory containing Tesseract language data files")
    ingest_parser.add_argument("--gpu", action="store_true", default=True,
                              help="Use GPU for OCR (only for EasyOCR, default: True)")
    ingest_parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                              help="Do not use GPU for OCR")
    
    # Build graph command
    build_parser = subparsers.add_parser("build-graph", help="Build knowledge graph from existing vector store")
    build_parser.add_argument("--company", help="Company ID to build graph for (uses active company if not specified)")
    build_parser.add_argument("--llm-model", help="LLM model to use for entity extraction")
    build_parser.add_argument("--embedding-model", help="Embedding model to use for vector store")
    build_parser.add_argument("--num-docs", type=int, default=50,
                             help="Number of documents to process from vector store (default: 50)")
    build_parser.add_argument("--query", help="Optional query to filter documents for graph construction")
    build_parser.add_argument("--visualize", action="store_true", default=False,
                             help="Visualize the resulting knowledge graph")
    build_parser.add_argument("--max-nodes", type=int, default=50,
                             help="Maximum number of nodes to display in visualization (default: 50)")
    
    # Visualize graph command
    viz_parser = subparsers.add_parser("visualize-graph", help="Visualize the knowledge graph")
    viz_parser.add_argument("--company", help="Company ID to visualize graph for (uses active company if not specified)")
    viz_parser.add_argument("--output", help="Output file path (default: knowledge_graph.png)")
    viz_parser.add_argument("--max-nodes", type=int, default=50,
                           help="Maximum number of nodes to display (default: 50)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the documents using GraphRAG")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument("--company", help="Company ID to query from (uses active company if not specified)")
    query_parser.add_argument("--raw-chunks", action="store_true", help="Return raw chunks without LLM processing")
    query_parser.add_argument("--num-chunks", type=int, help="Number of chunks to return (default from system settings)")
    query_parser.add_argument("--llm-model", help="Override LLM model for this query")
    query_parser.add_argument("--embedding-model", help="Override embedding model for this query")
    query_parser.add_argument("--temperature", type=float, help="Override temperature for this query")
    query_parser.add_argument("--template", help="Path to a custom prompt template file")
    query_parser.add_argument("--retrieval-method", choices=["vector", "graph", "hybrid"], default="hybrid",
                             help="Retrieval method to use (default: hybrid)")
    query_parser.add_argument("--num-hops", type=int, default=1,
                             help="Maximum number of hops for graph traversal (default: 1)")
    query_parser.add_argument("--explain", action="store_true", default=False,
                             help="Generate visual explanation of the query path")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_documents(args)
    elif args.command == "query":
        answer_question(args)
    elif args.command == "company":
        company_management(args)
    elif args.command == "model":
        model_management(args)
    elif args.command == "build-graph":
        build_graph(args)
    elif args.command == "visualize-graph":
        visualize_knowledge_graph(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
