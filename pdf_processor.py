#!/usr/bin/env python3
"""
PDF Processor for Financial Documents
Handles PDF extraction and processing for the Financial RAG system
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import re

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print(" Error: PDF processing libraries not found.")
    print("Please install: pip install PyPDF2 pdfplumber")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialDocument:
    """Financial document with extracted content"""
    file_name: str
    file_path: str
    text_content: str
    total_pages: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FinancialPDFProcessor:
    """Advanced PDF processor for financial documents"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        
        # Financial document patterns for better extraction
        self.financial_patterns = {
            'currency': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+\.?\d*%',
            'dates': r'\b(?:Q[1-4]|quarter|fiscal year|FY)\s+\d{4}\b',
            'financial_terms': r'\b(?:revenue|profit|loss|earnings|ebitda|cash flow|margin)\b'
        }
    
    def process_pdf_file(self, file_path: str) -> Optional[FinancialDocument]:
        """Process a single PDF file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            logger.info(f"Processing PDF: {file_path.name}")
            
            # Try pdfplumber first (better for financial documents)
            text_content = self._extract_with_pdfplumber(file_path)
            
            # Fallback to PyPDF2 if pdfplumber fails
            if not text_content or len(text_content.strip()) < 100:
                logger.warning(f"pdfplumber extraction poor, trying PyPDF2 for {file_path.name}")
                text_content = self._extract_with_pypdf2(file_path)
            
            if not text_content or len(text_content.strip()) < 50:
                logger.error(f"Failed to extract meaningful content from {file_path.name}")
                return None
            
            # Get page count
            total_pages = self._get_page_count(file_path)
            
            # Clean and enhance text
            cleaned_text = self._clean_text(text_content)
            
            # Create document object
            doc = FinancialDocument(
                file_name=file_path.name,
                file_path=str(file_path),
                text_content=cleaned_text,
                total_pages=total_pages,
                metadata={
                    'extraction_method': 'pdfplumber',
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'financial_indicators': self._analyze_financial_content(cleaned_text)
                }
            )
            
            logger.info(f"Successfully processed {file_path.name}: {total_pages} pages, {len(cleaned_text):,} chars")
            return doc
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_pdf_directory(self, directory_path: str) -> List[FinancialDocument]:
        """Process all PDF files in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        documents = []
        for pdf_file in pdf_files:
            doc = self.process_pdf_file(pdf_file)
            if doc:
                documents.append(doc)
        
        logger.info(f"Successfully processed {len(documents)} out of {len(pdf_files)} PDF files")
        return documents
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber (better for tables and financial data)"""
        try:
            text_parts = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # Convert table to text
                                table_text = self._table_to_text(table)
                                if table_text:
                                    text_parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {file_path.name}: {e}")
                        continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path.name}: {e}")
            return ""
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {file_path.name}: {e}")
                        continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path.name}: {e}")
            return ""
    
    def _get_page_count(self, file_path: Path) -> int:
        """Get the number of pages in the PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except:
            try:
                with pdfplumber.open(file_path) as pdf:
                    return len(pdf.pages)
            except:
                return 0
    
    def _table_to_text(self, table) -> str:
        """Convert table data to readable text"""
        if not table:
            return ""
        
        try:
            text_lines = []
            for row in table:
                if row and any(cell for cell in row if cell):  # Skip empty rows
                    # Clean and join cells
                    clean_cells = [str(cell).strip() if cell else "" for cell in row]
                    if any(clean_cells):  # Only add non-empty rows
                        text_lines.append(" | ".join(clean_cells))
            
            return "\n".join(text_lines)
        except Exception as e:
            logger.warning(f"Error converting table to text: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        # Remove page headers/footers patterns
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        
        # Normalize financial data
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # Fix "$  123" to "$123"
        text = re.sub(r'(\d)\s+%', r'\1%', text)  # Fix "12 %" to "12%"
        
        return text.strip()
    
    def _analyze_financial_content(self, text: str) -> dict:
        """Analyze text for financial content indicators"""
        analysis = {}
        
        for pattern_name, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis[pattern_name] = len(matches)
        
        # Calculate financial relevance score
        total_indicators = sum(analysis.values())
        word_count = len(text.split())
        analysis['financial_density'] = total_indicators / word_count if word_count > 0 else 0
        analysis['total_financial_indicators'] = total_indicators
        
        return analysis

# Example usage and testing
def main():
    """Example usage"""
    processor = FinancialPDFProcessor()
    
    # Test with a single file
    test_file = "sample_financial_report.pdf"
    if Path(test_file).exists():
        doc = processor.process_pdf_file(test_file)
        if doc:
            print(f" Processed: {doc.file_name}")
            print(f" Pages: {doc.total_pages}")
            print(f" Characters: {len(doc.text_content):,}")
            print(f" Financial indicators: {doc.metadata['financial_indicators']['total_financial_indicators']}")
        else:
            print(" Failed to process file")
    else:
        print(f" Test file {test_file} not found")
    
    # Test with directory
    test_dir = "./pdf_documents"
    if Path(test_dir).exists():
        docs = processor.process_pdf_directory(test_dir)
        print(f" Processed {len(docs)} documents from {test_dir}")
    else:
        print(f"ℹ  Test directory {test_dir} not found")

if __name__ == "__main__":
    main()