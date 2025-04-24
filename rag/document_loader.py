"""
Document loader module for RAG system with OCR support and configurable chunking
"""
import os
import tempfile
import pandas as pd
from typing import List, Set, Optional, Dict, Any, Union
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
from .config import SystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx', '.json', '.html'}

def is_supported_file(file_path: str) -> bool:
    """
    Check if a file is supported for ingestion
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is supported, False otherwise
    """
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in SUPPORTED_EXTENSIONS

def scan_directory(directory_path: str, recursive: bool = True) -> List[str]:
    """
    Scan a directory for supported files
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        List of file paths
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Not a directory: {directory_path}")
    
    supported_files = []
    
    # Walk through directory
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_supported_file(file_path):
                    supported_files.append(file_path)
    else:
        # Non-recursive scan
        for item in os.listdir(directory_path):
            file_path = os.path.join(directory_path, item)
            if os.path.isfile(file_path) and is_supported_file(file_path):
                supported_files.append(file_path)
    
    return supported_files

def needs_ocr(pdf_path: str) -> bool:
    """
    Check if a PDF needs OCR processing by looking for image content or lack of text
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        True if the PDF needs OCR, False otherwise
    """
    try:
        doc = fitz.open(pdf_path)
        total_text = 0
        has_images = False
        
        # Check first few pages (up to 5)
        pages_to_check = min(5, len(doc))
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # Check if page has text
            text = page.get_text()
            total_text += len(text)
            
            # Check for images
            img_list = page.get_images(full=True)
            if len(img_list) > 0:
                has_images = True
        
        doc.close()
        
        # If there are images and little text, OCR might be needed
        if has_images and total_text < 100 * pages_to_check:
            return True
        
        # If there's almost no text, OCR might be needed
        if total_text < 50 * pages_to_check:
            return True
            
        return False
    except Exception as e:
        logger.warning(f"Error checking if PDF needs OCR: {e}. Assuming regular PDF.")
        return False

def convert_xlsx_to_csv(xlsx_path: str) -> str:
    """
    Convert an Excel (XLSX) file to CSV format
    
    Args:
        xlsx_path: Path to the Excel file
        
    Returns:
        Path to the temporary CSV file
    """
    try:
        logger.info(f"Converting Excel file {xlsx_path} to CSV")
        
        # Read the Excel file with pandas
        df = pd.read_excel(xlsx_path)
        
        # Create a temporary file for the CSV output
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_csv_path = temp_file.name
        
        # Write the dataframe to CSV
        df.to_csv(temp_csv_path, index=False)
        logger.info(f"Converted Excel file to CSV: {temp_csv_path}")
        
        return temp_csv_path
    except Exception as e:
        logger.error(f"Error converting Excel file to CSV: {e}")
        raise

def load_document(file_path: str, ocr_enabled: bool = False, 
                 ocr_options: Optional[Dict[str, Any]] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None) -> List:
    """
    Load a document from a file path and split it into chunks
    
    Args:
        file_path: Path to the document
        ocr_enabled: Whether to enable OCR for PDFs with images
        ocr_options: OCR options including engine, language, etc.
        chunk_size: Size of chunks in characters (uses system setting if None)
        chunk_overlap: Overlap between chunks in characters (uses system setting if None)
        
    Returns:
        List of document chunks
    """
    _, file_extension = os.path.splitext(file_path)
    
    # Set default OCR options if not provided
    if ocr_options is None:
        ocr_options = {
            'engine': 'tesseract',
            'lang': 'eng',
            'dpi': 300,
            'use_gpu': True
        }
    
    # Process XLSX files - convert to CSV first
    if file_extension.lower() == '.xlsx':
        try:
            # Convert XLSX to CSV
            temp_csv_path = convert_xlsx_to_csv(file_path)
            
            # Load the CSV file
            loader = TextLoader(temp_csv_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(temp_csv_path)
            
            logger.info(f"Loaded Excel file {file_path} as CSV")
        except Exception as e:
            logger.error(f"Failed to process Excel file: {e}")
            raise
    
    # Process PDF files
    elif file_extension.lower() == '.pdf':
        # Check if PDF needs OCR and OCR is enabled
        if ocr_enabled and needs_ocr(file_path):
            logger.info(f"PDF appears to contain images or scanned content. Using OCR.")
            
            # Get OCR text
            from .ocr import convert_pdf_to_text
            
            # Create a temporary file for the OCR output
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Perform OCR
                logger.info(f"Processing {file_path} with OCR...")
                convert_pdf_to_text(
                    pdf_path=file_path,
                    output_path=temp_path,
                    engine=ocr_options.get('engine', 'tesseract'),
                    lang=ocr_options.get('lang', 'eng'),
                    dpi=ocr_options.get('dpi', 300),
                    use_gpu=ocr_options.get('use_gpu', True),
                    tesseract_cmd=ocr_options.get('tesseract_cmd', None),
                    tessdata_dir=ocr_options.get('tessdata_dir', None)
                )
                
                # Load the OCR text
                loader = TextLoader(temp_path)
                documents = loader.load()
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                logger.error(f"OCR processing failed: {e}. Falling back to regular PDF loading.")
                # Clean up temporary file if it exists
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
                # Fallback to regular PDF loading
                loader = PyPDFLoader(file_path)
                documents = loader.load()
        else:
            # Regular PDF loading
            loader = PyPDFLoader(file_path)
            documents = loader.load()
    
    # Process other file types
    elif file_extension.lower() == '.docx':
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    elif file_extension.lower() in ['.txt', '.md', '.csv', '.json', '.html']:
        loader = TextLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Get chunk settings from system config if not provided
    if chunk_size is None or chunk_overlap is None:
        sys_config = SystemConfig()
        chunk_settings = sys_config.get_chunk_settings()
        
        if chunk_size is None:
            chunk_size = chunk_settings.get("chunk_size", 1000)
        
        if chunk_overlap is None:
            chunk_overlap = chunk_settings.get("chunk_overlap", 200)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split document into {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}")
    return chunks
