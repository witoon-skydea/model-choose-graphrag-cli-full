# Model-Choose GraphRAG CLI Full

A powerful GraphRAG (Graph-enhanced Retrieval-Augmented Generation) Command Line Interface tool with customizable model selection and multi-company support. Built on LangChain and Ollama, this tool enables advanced document processing, knowledge graph creation, OCR, and Q&A capabilities with flexible model configurations.

## Features

- **Knowledge Graph Integration**: Build and query knowledge graphs extracted from documents
- **Hybrid Retrieval**: Combine vector search with knowledge graph exploration
- **Entity Extraction**: Automatically extract entities and relationships from documents
- **Graph Visualization**: Visualize knowledge graphs and query paths
- **Flexible Model Selection**: Choose different LLM and embedding models for different tasks or companies
- **Multi-Company Support**: Organize your document databases by company or project
- **OCR Capabilities**: Process scanned PDFs and images with Tesseract or EasyOCR
- **Document Processing**: Support for PDF, DOCX, TXT, MD, CSV, XLSX, JSON, and HTML files
- **Configurable Chunking**: Customize chunk size and overlap for different document types
- **Custom Prompts**: Use custom prompt templates for specialized queries
- **System-wide Settings**: Set default models, temperatures, and other parameters
- **Support for Thai Language**: Thai-specific entity extraction for better graph building with Thai content

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed locally
- Tesseract OCR (optional, for OCR functionality)

### Setup

1. Clone the repository
2. Run the setup script:

```bash
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install required dependencies
- Check for required Ollama models and pull them if needed
- Set up necessary directories

## Usage

Use the `run.sh` script to interact with the application:

```bash
./run.sh --help  # Show help information
```

### Model Management

```bash
# List available models
./run.sh model --list-all

# Show current system settings
./run.sh model --show-settings

# Set default system settings
./run.sh model --set-defaults --default-llm phi3 --temperature 0.5
```

### Company Management

```bash
# List all companies
./run.sh company --list

# Add a new company
./run.sh company --add --id research --name "Research Department" --llm-model llama3:70b

# Set active company
./run.sh company --set-active --id research

# Show active company
./run.sh company --show-active

# Set models for a company
./run.sh company --set-models --id research --llm-model phi3 --embedding-model nomic
```

### Document Ingestion

```bash
# Ingest a document
./run.sh ingest path/to/file.pdf

# Ingest documents recursively from a directory
./run.sh ingest path/to/directory --recursive

# Ingest for a specific company
./run.sh ingest file.pdf --company research

# Use a specific embedding model
./run.sh ingest file.pdf --embedding-model nomic

# Enable OCR for PDFs with images
./run.sh ingest file.pdf --ocr

# OCR with specific options
./run.sh ingest file.pdf --ocr --ocr-engine easyocr --ocr-lang eng+tha

# Build knowledge graph during ingestion
./run.sh ingest file.pdf --build-graph

# Build and visualize knowledge graph
./run.sh ingest file.pdf --build-graph --visualize-graph
```

### Knowledge Graph Management

```bash
# Build knowledge graph from existing vector store
./run.sh build-graph

# Build graph for a specific company
./run.sh build-graph --company research

# Set number of documents to process
./run.sh build-graph --num-docs 100

# Filter documents by query for graph building
./run.sh build-graph --query "artificial intelligence"

# Visualize the knowledge graph
./run.sh build-graph --visualize

# Visualize existing knowledge graph
./run.sh visualize-graph

# Visualize with maximum nodes limit
./run.sh visualize-graph --max-nodes 100
```

### Querying

```bash
# Basic query
./run.sh query "What is in the document?"

# Query a specific company
./run.sh query "What is in the document?" --company research

# Use a specific LLM model
./run.sh query "What is in the document?" --llm-model phi3

# Set temperature
./run.sh query "What is in the document?" --temperature 0.3

# Set number of chunks to return
./run.sh query "What is in the document?" --num-chunks 6

# Return raw chunks without LLM processing
./run.sh query "What is in the document?" --raw-chunks

# Use a custom prompt template
./run.sh query "What is in the document?" --template path/to/template.txt

# Specify retrieval method (vector, graph, hybrid)
./run.sh query "What is in the document?" --retrieval-method hybrid

# Set number of hops for graph traversal
./run.sh query "What is in the document?" --num-hops 2

# Generate visual explanation of query path
./run.sh query "How are concepts X and Y related?" --explain
```

## Advanced Usage

### Knowledge Graph and GraphRAG

The system builds knowledge graphs by extracting entities and relationships from your documents. This enhances retrieval by capturing the connections between concepts:

- **Entity Extraction**: The system automatically extracts people, organizations, locations, and concepts from your documents
- **Relationship Building**: It identifies how these entities are connected
- **Graph Traversal**: When you query, the system can find relevant information by traversing relationships

For complex queries that involve connecting concepts, use the graph or hybrid retrieval method:

```bash
# Use pure graph-based retrieval
./run.sh query "How are X and Y related?" --retrieval-method graph

# Use hybrid retrieval (combining vector similarity and graph exploration)
./run.sh query "Tell me about the relationship between X and Y" --retrieval-method hybrid --num-hops 2

# Visualize how the system found the answer
./run.sh query "What's the connection between A and B?" --explain
```

For Thai language documents, the system uses specialized entity extraction to better handle Thai text:

```bash
# Build graph with Thai content
./run.sh ingest thai_documents.pdf --build-graph
```

### Custom Prompt Templates

You can create custom prompt templates to control how the LLM responds to queries. Create a text file with the following format:

```
You are an expert in finance. Use the following pieces of context and knowledge graph information to answer the user's question in a professional tone.

Context:
{context}

Knowledge Graph Information:
{graph_context}

User Question: {question}

Your Expert Answer:
```

Then use it with the `--template` flag:

```bash
./run.sh query "What are the financial results?" --template prompts/finance_expert.txt
```

### OCR Configuration

For documents with images or scanned PDFs, enable OCR:

```bash
./run.sh ingest scanned_document.pdf --ocr --ocr-engine tesseract --ocr-lang eng
```

Supported OCR engines:
- `tesseract`: Fast and widely supported
- `easyocr`: More accurate for certain languages but slower

## Supported Models

### LLM Models

- llama3:8b (default)
- llama3:70b
- phi3
- gemma:7b
- mistral
- codellama
- wizardcoder
- llava

### Embedding Models

- mxbai-embed-large (default)
- nomic-embed-text
- llama3:8b
- phi3

You can also use any other model available in Ollama by specifying its name directly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
