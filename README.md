# Model-Choose GraphRAG CLI Full

A powerful GraphRAG (Graph-enhanced Retrieval-Augmented Generation) command-line tool with customizable model selection and multi-company support. Built on LangChain and Ollama, this tool enhances traditional RAG systems with knowledge graph capabilities for more robust document understanding and question answering.

## Features

- **Knowledge Graph Integration**: 
  - Extract entities and relationships from documents automatically
  - Build rich knowledge graphs to represent document connections
  - Thai language support for entity extraction

- **Advanced Retrieval Methods**:
  - Vector search for traditional similarity matching
  - Graph search for relationship-based retrieval
  - Hybrid search combining both approaches for optimal results

- **Visualization**:
  - Visualize knowledge graphs to explore document connections
  - Generate visual explanations of query reasoning paths

- **Model Flexibility**:
  - Choose different LLM and embedding models for different tasks
  - Support for multiple companies or projects with separate configurations

- **Document Processing**:
  - Support for PDF, DOCX, TXT, MD, CSV, XLSX, JSON, and HTML files
  - OCR capabilities for scanned documents using Tesseract or EasyOCR
  - Configurable chunking strategies

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

## Usage Examples

### Building Knowledge Graphs

```bash
# Ingest documents and build knowledge graph
./run.sh ingest path/to/documents/ --build-graph

# Build graph from existing vector store
./run.sh build-graph --company research

# Visualize the knowledge graph
./run.sh visualize-graph
```

### GraphRAG Querying

```bash
# Query using hybrid retrieval (vector + graph)
./run.sh query "What is the relationship between X and Y?" --retrieval-method hybrid

# Query using only graph-based retrieval
./run.sh query "How are these concepts connected?" --retrieval-method graph --num-hops 2

# Generate visual explanation of reasoning
./run.sh query "Explain the connection between A and B" --explain
```

### Company Management

```bash
# Add a new company with specific models
./run.sh company --add --id research --name "Research Department" --llm-model llama3:70b

# Set active company
./run.sh company --set-active --id research
```

## Architecture

The system combines traditional vector-based RAG with knowledge graph capabilities:

1. **Document Processing**: Parse and chunk documents with optional OCR
2. **Entity Extraction**: Extract entities and relationships using LLMs
3. **Graph Building**: Construct a knowledge graph representing the document content
4. **Hybrid Retrieval**: Combine vector similarity with graph traversal
5. **Response Generation**: Use retrieved context with graph information for enhanced responses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
