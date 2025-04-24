#!/bin/bash
# Run script for model-choose-graphrag-cli-full

# Colors and formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Check if help was requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo -e "\n${BLUE}${BOLD}📚 Model-Choose GraphRAG CLI Full - Help${NC}\n"
    echo -e "${YELLOW}Available commands:${NC}"
    echo -e "\n${BOLD}Model Management:${NC}"
    echo -e "  ${BLUE}./run.sh model --list-llm${NC}: List available LLM models"
    echo -e "  ${BLUE}./run.sh model --list-embeddings${NC}: List available embedding models"
    echo -e "  ${BLUE}./run.sh model --list-all${NC}: List all available models"
    echo -e "  ${BLUE}./run.sh model --show-settings${NC}: Show current system settings"
    echo -e "  ${BLUE}./run.sh model --set-defaults --default-llm llama3:8b --temperature 0.7${NC}: Set default system settings"
    
    echo -e "\n${BOLD}Company Management:${NC}"
    echo -e "  ${BLUE}./run.sh company --list${NC}: List all companies"
    echo -e "  ${BLUE}./run.sh company --add --id mycompany --name \"My Company\"${NC}: Add a new company"
    echo -e "  ${BLUE}./run.sh company --remove --id mycompany${NC}: Remove a company"
    echo -e "  ${BLUE}./run.sh company --set-active --id mycompany${NC}: Set active company"
    echo -e "  ${BLUE}./run.sh company --show-active${NC}: Show active company"
    echo -e "  ${BLUE}./run.sh company --set-models --id mycompany --llm-model phi3${NC}: Set models for a company"
    
    echo -e "\n${BOLD}Document Ingestion:${NC}"
    echo -e "  ${BLUE}./run.sh ingest path/to/file.pdf${NC}: Ingest a document"
    echo -e "  ${BLUE}./run.sh ingest path/to/directory --recursive${NC}: Ingest documents recursively"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --company mycompany${NC}: Ingest for specific company"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --embedding-model nomic${NC}: Use specific embedding model"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --ocr${NC}: Enable OCR for PDF files"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --ocr --ocr-engine easyocr --ocr-lang eng+tha${NC}: OCR with specific options"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --build-graph${NC}: Build knowledge graph during ingestion"
    echo -e "  ${BLUE}./run.sh ingest file.pdf --build-graph --visualize-graph${NC}: Build and visualize graph"
    
    echo -e "\n${BOLD}Knowledge Graph:${NC}"
    echo -e "  ${BLUE}./run.sh build-graph${NC}: Build knowledge graph from existing vector store"
    echo -e "  ${BLUE}./run.sh build-graph --company mycompany${NC}: Build graph for specific company"
    echo -e "  ${BLUE}./run.sh build-graph --num-docs 100${NC}: Set number of documents to process"
    echo -e "  ${BLUE}./run.sh build-graph --query \"topic\"${NC}: Filter documents by query for graph building"
    echo -e "  ${BLUE}./run.sh build-graph --visualize${NC}: Visualize the resulting knowledge graph"
    echo -e "  ${BLUE}./run.sh visualize-graph${NC}: Visualize existing knowledge graph"
    echo -e "  ${BLUE}./run.sh visualize-graph --max-nodes 100${NC}: Set maximum number of nodes to display"
    
    echo -e "\n${BOLD}Querying:${NC}"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\"${NC}: Query the document"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --company mycompany${NC}: Query specific company"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --llm-model phi3${NC}: Use specific LLM model"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --embedding-model nomic${NC}: Use specific embedding model"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --temperature 0.5${NC}: Set temperature"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --num-chunks 6${NC}: Set number of chunks to return"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --raw-chunks${NC}: Return raw chunks without LLM processing"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --template path/to/template.txt${NC}: Use custom prompt template"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --retrieval-method hybrid${NC}: Set retrieval method (vector, graph, hybrid)"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --num-hops 2${NC}: Set number of hops for graph traversal"
    echo -e "  ${BLUE}./run.sh query \"What is in the document?\" --explain${NC}: Generate visual explanation of query path"
    
    echo -e "\n${BOLD}Examples:${NC}"
    echo -e "  ${BLUE}./run.sh model --list-all${NC}"
    echo -e "  ${BLUE}./run.sh company --add --id research --name \"Research Department\" --llm-model llama3:70b${NC}"
    echo -e "  ${BLUE}./run.sh ingest ./data/research_papers/ --company research --recursive --build-graph${NC}"
    echo -e "  ${BLUE}./run.sh build-graph --company research --visualize${NC}"
    echo -e "  ${BLUE}./run.sh query \"Summarize the key findings in the research papers\" --company research --retrieval-method hybrid${NC}"
    echo -e "  ${BLUE}./run.sh query \"How are concepts X and Y related?\" --retrieval-method graph --explain${NC}"
    
    exit 0
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run ./setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run Python script
python3 main.py "$@"
