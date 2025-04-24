"""
OCR functionality for PDF processing.
Adapted from pdf-ocr-cli project.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import numpy as np
import io
from typing import List, Tuple, Union, Optional, Dict, Any
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRBase:
    """Base class for OCR engines."""
    
    def __init__(self, lang: str = 'eng', dpi: int = 300):
        """
        Initialize OCR engine.
        
        Args:
            lang: Language code(s) for OCR (e.g., 'tha' for Thai, 'eng+tha' for English and Thai)
            dpi: DPI setting for PDF to image conversion
        """
        self.lang = lang
        self.dpi = dpi
        
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image before OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        # Default implementation: return the original image
        return image


class TesseractOCR(OCRBase):
    """OCR engine using Tesseract."""
    
    def __init__(self, lang: str = 'eng', dpi: int = 300, tesseract_cmd: Optional[str] = None, tessdata_dir: Optional[str] = None):
        """
        Initialize Tesseract OCR engine.
        
        Args:
            lang: Language code(s) for OCR (e.g., 'eng' for English, 'eng+tha' for English and Thai)
            dpi: DPI setting for PDF to image conversion
            tesseract_cmd: Path to Tesseract executable (if not in PATH)
            tessdata_dir: Path to tessdata directory containing language data files
        """
        super().__init__(lang, dpi)
        
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            if tesseract_cmd:
                self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                
            # Set TESSDATA_PREFIX environment variable if tessdata_dir is provided
            if tessdata_dir:
                os.environ['TESSDATA_PREFIX'] = tessdata_dir
                logger.info(f"Set TESSDATA_PREFIX to {tessdata_dir}")
        except ImportError:
            raise ImportError("pytesseract is not installed. Install it using: pip install pytesseract")
        
        # Check if Tesseract is available
        try:
            self.pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Could not find Tesseract executable. Make sure it's installed and in PATH, or set tesseract_cmd. Error: {e}")
            
        # Check if the specified language is available
        try:
            available_langs = self.pytesseract.get_languages()
            lang_codes = lang.split('+')
            
            for lang_code in lang_codes:
                if lang_code not in available_langs:
                    tesseract_path = self.pytesseract.pytesseract.tesseract_cmd
                    tessdata_prefix = os.environ.get('TESSDATA_PREFIX', '(not set)')
                    
                    raise RuntimeError(
                        f"Language '{lang_code}' not found in available languages: {available_langs}.\n\n"
                        f"Installation Instructions:\n"
                        f"- macOS: brew install tesseract-lang\n"
                        f"- Ubuntu/Debian: sudo apt-get install tesseract-ocr-{lang_code}\n"
                        f"- Windows: Download language data from https://github.com/tesseract-ocr/tessdata/ and place in tessdata directory\n\n"
                        f"Current configuration:\n"
                        f"- Tesseract path: {tesseract_path}\n"
                        f"- TESSDATA_PREFIX: {tessdata_prefix}\n"
                    )
        except Exception as e:
            logger.warning(f"Could not verify language availability: {e}")
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using Tesseract.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        image = self.preprocess_image(image)
        
        try:
            text = self.pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""


class EasyOCR(OCRBase):
    """OCR engine using EasyOCR."""
    
    def __init__(self, lang: str = 'en', dpi: int = 300, use_gpu: bool = True):
        """
        Initialize EasyOCR engine.
        
        Args:
            lang: Language code(s) for OCR (e.g., 'en' for English, 'th' for Thai, 'th+en' for Thai and English)
            dpi: DPI setting for PDF to image conversion
            use_gpu: Whether to use GPU for OCR
        """
        super().__init__(lang, dpi)
        self.use_gpu = use_gpu
        
        # Convert language codes from Tesseract format to EasyOCR format
        # Tesseract uses 'tha+eng', EasyOCR uses ['th', 'en']
        lang_map = {
            'tha': ['th'],
            'eng': ['en'],
            'tha+eng': ['th', 'en'],
            'eng+tha': ['en', 'th']
        }
        
        if lang in lang_map:
            self.easyocr_lang = lang_map[lang]
        else:
            # Handle custom language combinations
            lang_parts = lang.split('+')
            self.easyocr_lang = []
            for part in lang_parts:
                if part == 'tha':
                    self.easyocr_lang.append('th')
                elif part == 'eng':
                    self.easyocr_lang.append('en')
                else:
                    # For other languages, assume the code format is the same
                    self.easyocr_lang.append(part)
        
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages: {self.easyocr_lang}")
            self.reader = easyocr.Reader(self.easyocr_lang, gpu=self.use_gpu)
        except ImportError:
            raise ImportError("easyocr is not installed. Install it using: pip install easyocr")
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using EasyOCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        image = self.preprocess_image(image)
        
        try:
            # Convert PIL Image to numpy array
            img_np = np.array(image)
            
            # Perform OCR
            result = self.reader.readtext(img_np, detail=0, paragraph=True)
            
            # Join results
            text = "\n".join(result)
            return text
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""


def get_ocr_engine(engine: str, lang: str = 'eng', dpi: int = 300, **kwargs) -> OCRBase:
    """
    Factory function to get an OCR engine instance.
    
    Args:
        engine: OCR engine name ('tesseract' or 'easyocr')
        lang: Language code(s) for OCR
        dpi: DPI setting for PDF to image conversion
        **kwargs: Additional arguments for the OCR engine
        
    Returns:
        OCR engine instance
    """
    if engine.lower() == 'tesseract':
        # Check system paths for tessdata directory if not provided
        if 'tessdata_dir' not in kwargs:
            potential_paths = [
                "/usr/share/tessdata",
                "/usr/local/share/tessdata",
                "/opt/homebrew/share/tessdata",
                "/opt/tesseract/share/tessdata",
                "/usr/share/tesseract-ocr/tessdata",
                "C:\\Program Files\\Tesseract-OCR\\tessdata"
            ]
            
            for path in potential_paths:
                if os.path.isdir(path) and os.path.exists(os.path.join(path, f"{lang.split('+')[0]}.traineddata")):
                    logger.info(f"Found tessdata directory at {path}")
                    kwargs['tessdata_dir'] = path
                    break
        
        return TesseractOCR(lang=lang, dpi=dpi, **kwargs)
    elif engine.lower() == 'easyocr':
        return EasyOCR(lang=lang, dpi=dpi, **kwargs)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}")


def convert_pdf_to_text(
    pdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    engine: str = 'tesseract',
    lang: str = 'eng',
    dpi: int = 300,
    **kwargs
) -> str:
    """
    Convert a PDF file to text using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text to (optional)
        engine: OCR engine to use ('tesseract' or 'easyocr')
        lang: Language code(s) for OCR
        dpi: DPI setting for PDF to image conversion
        **kwargs: Additional arguments for the OCR engine
        
    Returns:
        Extracted text
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Get OCR engine
    ocr_engine = get_ocr_engine(engine, lang, dpi, **kwargs)
    
    # Open PDF file
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    all_text = ""
    num_pages = len(doc)
    logger.info(f"Processing {num_pages} pages")
    
    for i, page in tqdm(enumerate(doc), total=num_pages, desc="OCR Progress"):
        logger.info(f"Processing page {i+1}/{num_pages}")
        
        # Render page to image
        pix = page.get_pixmap(dpi=dpi)
        
        # Convert to PIL Image
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        
        # Perform OCR
        text = ocr_engine.extract_text(img)
        
        # Add page marker
        all_text += f"--- Page {i+1} ---\n{text}\n\n"
    
    # Close PDF file
    doc.close()
    
    # Save to file if output_path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        
        logger.info(f"Extracted text saved to: {output_path}")
    
    return all_text
