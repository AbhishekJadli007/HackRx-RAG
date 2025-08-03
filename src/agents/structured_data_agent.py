from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import camelot
import pdfplumber
from PyPDF2 import PdfReader
import pandas as pd
import re
import os
import traceback

class StructuredDataExtractionAgent(BaseAgent):
    """Agent responsible for extracting structured data (tables and chat) from PDFs"""
    
    def __init__(self):
        super().__init__("structured_data_extraction", "structured_data")
        
        # Common chat patterns for different formats
        self.chat_patterns = {
            "timestamp_speaker": r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*([^:]+):\s*(.*)",
            "speaker_colon": r"^([^:]+):\s*(.*)",
            "bracketed_speaker": r"^\[([^\]]+)\]:\s*(.*)",
            "user_admin": r"^(User|Admin|Bot|Agent|Customer|Support):\s*(.*)",
            "date_timestamp": r"^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)"
        }
        
        # Table detection keywords
        self.table_keywords = [
            "table", "chart", "data", "figures", "statistics", "metrics",
            "performance", "results", "summary", "comparison", "analysis"
        ]
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF extraction requests for tables and chat data"""
        self.log_activity("processing_extraction", input_data)
        
        # Health check: if only 'query' is present and it's 'health check', return success
        if input_data.get("query", "").lower() == "health check":
            return self.create_response(True, data={"status": "ok", "agent": "structured_data_extraction"})

        if not self.validate_input(input_data, ["pdf_path", "data_type"]):
            return self.create_response(False, error="Missing required fields: pdf_path, data_type")
        
        pdf_path = input_data["pdf_path"]
        data_type = input_data["data_type"]
        pages = input_data.get("pages", "all")
        
        if not os.path.exists(pdf_path):
            return self.create_response(False, error=f"PDF file not found: {pdf_path}")
        
        try:
            if data_type == "table":
                result = self._extract_tables(pdf_path, pages)
            elif data_type == "chat":
                result = self._extract_chat(pdf_path, pages)
            else:
                return self.create_response(False, error=f"Unsupported data type: {data_type}")
            
            self.log_activity("extraction_completed", {"data_type": data_type, "pages": pages})
            return self.create_response(True, data=result)
            
        except Exception as e:
            self.log_activity("extraction_error", {"error": str(e), "traceback": traceback.format_exc()})
            return self.create_response(False, error=f"Error during extraction: {str(e)}")
    
    def _extract_tables(self, pdf_path: str, pages: str = "all") -> Dict[str, Any]:
        """Extract tables from PDF using multiple methods"""
        result = {
            "tables": [],
            "extraction_method": "none",
            "total_tables": 0
        }
        
        # Try Camelot first (best for text-based PDFs with clear table structures)
        try:
            camelot_tables = self._extract_with_camelot(pdf_path, pages)
            if camelot_tables:
                result["tables"] = camelot_tables
                result["extraction_method"] = "camelot"
                result["total_tables"] = len(camelot_tables)
                return result
        except Exception as e:
            self.log_activity("camelot_failed", {"error": str(e)})
        
        # Fallback to pdfplumber
        try:
            pdfplumber_tables = self._extract_with_pdfplumber(pdf_path, pages)
            if pdfplumber_tables:
                result["tables"] = pdfplumber_tables
                result["extraction_method"] = "pdfplumber"
                result["total_tables"] = len(pdfplumber_tables)
                return result
        except Exception as e:
            self.log_activity("pdfplumber_failed", {"error": str(e)})
        
        # If no tables found, return empty result
        result["message"] = "No tables found using available extraction methods."
        return result
    
    def _extract_with_camelot(self, pdf_path: str, pages: str) -> List[Dict[str, Any]]:
        """Extract tables using Camelot"""
        tables = []
        
        # Try lattice method first (for tables with clear borders)
        try:
            lattice_tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice', suppress_stdout=True)
            if lattice_tables:
                for i, table in enumerate(lattice_tables):
                    if table.df.shape[0] > 1:  # Ensure table has data
                        tables.append({
                            "page": table.page,
                            "table_number": i + 1,
                            "data": table.df.to_dict(orient='records'),
                            "accuracy": table.accuracy,
                            "whitespace": table.whitespace,
                            "method": "lattice"
                        })
        except Exception as e:
            self.log_activity("camelot_lattice_failed", {"error": str(e)})
        
        # If no lattice tables, try stream method
        if not tables:
            try:
                stream_tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream', suppress_stdout=True)
                if stream_tables:
                    for i, table in enumerate(stream_tables):
                        if table.df.shape[0] > 1:
                            tables.append({
                                "page": table.page,
                                "table_number": i + 1,
                                "data": table.df.to_dict(orient='records'),
                                "accuracy": table.accuracy,
                                "whitespace": table.whitespace,
                                "method": "stream"
                            })
            except Exception as e:
                self.log_activity("camelot_stream_failed", {"error": str(e)})
        
        return tables
    
    def _extract_with_pdfplumber(self, pdf_path: str, pages: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = self._get_pages_to_process(pdf, pages)
                
                for page_num in pages_to_process:
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    for i, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:
                            # Use first row as header if it looks like headers
                            if self._looks_like_header(table_data[0]):
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            else:
                                df = pd.DataFrame(table_data)
                            
                            tables.append({
                                "page": page_num + 1,
                                "table_number": i + 1,
                                "data": df.to_dict(orient='records'),
                                "method": "pdfplumber"
                            })
        except Exception as e:
            self.log_activity("pdfplumber_error", {"error": str(e)})
        
        return tables
    
    def _extract_chat(self, pdf_path: str, pages: str = "all") -> Dict[str, Any]:
        """Extract chat/conversation data from PDF"""
        result = {
            "chat_segments": [],
            "total_segments": 0,
            "extraction_method": "heuristic"
        }
        
        try:
            reader = PdfReader(pdf_path)
            pages_to_process = self._get_pages_to_process(reader, pages)
            
            all_chat_lines = []
            for page_num in pages_to_process:
                page = reader.pages[page_num]
                text = page.extract_text()
                
                # Extract chat lines using multiple patterns
                chat_lines = self._extract_chat_lines(text, page_num + 1)
                all_chat_lines.extend(chat_lines)
            
            if all_chat_lines:
                result["chat_segments"] = all_chat_lines
                result["total_segments"] = len(all_chat_lines)
            else:
                result["message"] = "No chat segments found using heuristic patterns."
                
        except Exception as e:
            self.log_activity("chat_extraction_error", {"error": str(e)})
            result["message"] = f"Error extracting chat: {str(e)}"
        
        return result
    
    def _extract_chat_lines(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract chat lines using regex patterns"""
        chat_lines = []
        lines = text.split('\n')
        
        current_speaker = None
        current_message = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Try different chat patterns
            speaker = None
            message = None
            
            for pattern_name, pattern in self.chat_patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if pattern_name == "timestamp_speaker":
                        timestamp, speaker, message = match.groups()
                    elif pattern_name == "date_timestamp":
                        date, time, speaker, message = match.groups()
                        timestamp = f"{date} {time}"
                    else:
                        speaker, message = match.groups()
                        timestamp = None
                    break
            
            if speaker and message:
                # Save previous message if exists
                if current_speaker and current_message:
                    chat_lines.append({
                        "page": page_num,
                        "line": line_num + 1,
                        "speaker": current_speaker,
                        "message": " ".join(current_message),
                        "timestamp": None
                    })
                
                # Start new message
                current_speaker = speaker
                current_message = [message]
            elif current_speaker and line:
                # Continuation of previous message
                current_message.append(line)
        
        # Add the last message
        if current_speaker and current_message:
            chat_lines.append({
                "page": page_num,
                "line": len(lines),
                "speaker": current_speaker,
                "message": " ".join(current_message),
                "timestamp": None
            })
        
        return chat_lines
    
    def _get_pages_to_process(self, pdf_obj, pages: str) -> List[int]:
        """Convert pages string to list of page indices"""
        if pages == "all":
            return list(range(len(pdf_obj.pages)))
        
        try:
            if "-" in pages:
                start, end = map(int, pages.split("-"))
                return list(range(start - 1, end))
            elif "," in pages:
                return [int(p.strip()) - 1 for p in pages.split(",") if p.strip().isdigit()]
            else:
                return [int(pages) - 1]
        except:
            return [0]  # Default to first page if parsing fails
    
    def _looks_like_header(self, row: List[str]) -> bool:
        """Check if a row looks like a table header"""
        if not row:
            return False
        
        # Check if most cells contain short text (typical for headers)
        short_text_count = sum(1 for cell in row if cell and len(str(cell).strip()) < 20)
        return short_text_count >= len(row) * 0.7  # 70% should be short text
    
    def detect_data_type(self, query: str) -> str:
        """Detect if query is asking for tables or chat data"""
        query_lower = query.lower()
        
        # Check for table-related keywords
        if any(keyword in query_lower for keyword in self.table_keywords):
            return "table"
        
        # Check for chat-related keywords
        chat_keywords = ["chat", "conversation", "dialogue", "message", "speaker", "conversation"]
        if any(keyword in query_lower for keyword in chat_keywords):
            return "chat"
        
        return "text"  # Default to regular text retrieval

# Sample LLM prompt for table extraction:
# "Extract all tables from the PDF and display them in markdown table format. If the user asks for a specific table, show only that table. If possible, include the table number and page number." 