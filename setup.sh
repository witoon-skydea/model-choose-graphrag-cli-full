#!/bin/bash
# Setup script for model-choose-rag-cli-full

# Colors and formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "\n${BLUE}${BOLD}📚 Setting up Model-Choose RAG CLI Full...${NC}\n"

# Check if Python 3.8+ is installed
echo -e "${YELLOW}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Python 3.8 or higher is required. Found Python $PYTHON_VERSION.${NC}"
    exit 1
fi

echo -e "${GREEN}Python $PYTHON_VERSION found. Continuing...${NC}"

# Create and activate virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# Check for Tesseract OCR
echo -e "\n${YELLOW}Checking for Tesseract OCR...${NC}"
if ! command -v tesseract &> /dev/null; then
    echo -e "${RED}Tesseract OCR not found. OCR features will be limited.${NC}"
    echo -e "${YELLOW}To install Tesseract:${NC}"
    echo -e "  - ${BLUE}macOS:${NC} brew install tesseract tesseract-lang"
    echo -e "  - ${BLUE}Ubuntu/Debian:${NC} sudo apt-get install tesseract-ocr libtesseract-dev"
    echo -e "  - ${BLUE}Windows:${NC} Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
else
    TESSERACT_VERSION=$(tesseract --version | head -n 1)
    echo -e "${GREEN}$TESSERACT_VERSION found.${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p db/default
mkdir -p graph/default
mkdir -p data
mkdir -p companies
echo -e "${GREEN}Directories created.${NC}"

# Check for Ollama and suggest installation if not found
echo -e "\n${YELLOW}Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama not found. The application requires Ollama to function.${NC}"
    echo -e "${YELLOW}To install Ollama:${NC}"
    echo -e "  - Visit ${BLUE}https://ollama.com/download${NC} and follow the instructions for your OS."
    echo -e "  - After installation, run:"
    echo -e "    ${BLUE}ollama pull llama3:8b${NC} for the default LLM model"
    echo -e "    ${BLUE}ollama pull mxbai-embed-large${NC} for the default embedding model\n"
else
    OLLAMA_VERSION=$(ollama --version)
    echo -e "${GREEN}Ollama $OLLAMA_VERSION found.${NC}"
    
    # Check for required models
    echo -e "\n${YELLOW}Checking for required Ollama models...${NC}"
    
    # Get list of available models
    OLLAMA_MODELS=$(ollama list)
    
    # Check for LLM model
    if echo "$OLLAMA_MODELS" | grep -q "llama3:8b"; then
        echo -e "${GREEN}Default LLM model (llama3:8b) found.${NC}"
    else
        echo -e "${YELLOW}Default LLM model (llama3:8b) not found. Pulling...${NC}"
        ollama pull llama3:8b
    fi
    
    # Check for embedding model
    if echo "$OLLAMA_MODELS" | grep -q "mxbai-embed-large"; then
        echo -e "${GREEN}Default embedding model (mxbai-embed-large) found.${NC}"
    else
        echo -e "${YELLOW}Default embedding model (mxbai-embed-large) not found. Pulling...${NC}"
        ollama pull mxbai-embed-large
    fi
fi

# Set execution permissions
echo -e "\n${YELLOW}Setting execution permissions...${NC}"
chmod +x run.sh
chmod +x main.py
echo -e "${GREEN}Execution permissions set.${NC}"

echo -e "\n${GREEN}${BOLD}✅ Setup complete!${NC}"
echo -e "${BLUE}To run the application, use: ${BOLD}./run.sh${NC}"
echo -e "${BLUE}For help, use: ${BOLD}./run.sh --help${NC}\n"
echo -e "${YELLOW}Available commands:${NC}"
echo -e "  - ${BOLD}./run.sh model --list-all${NC}: List all available models"
echo -e "  - ${BOLD}./run.sh company --add --id mycompany --name \"My Company\"${NC}: Add a new company"
echo -e "  - ${BOLD}./run.sh ingest data/sample.pdf${NC}: Ingest a document"
echo -e "  - ${BOLD}./run.sh query \"What is in the document?\"${NC}: Query the document"
echo -e "${BLUE}For a complete list of commands, refer to the documentation.${NC}\n"
