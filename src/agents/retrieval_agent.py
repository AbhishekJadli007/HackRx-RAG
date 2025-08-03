from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import numpy as np
from difflib import SequenceMatcher

class AdaptiveRetrievalAgent(BaseAgent):
    """Agent responsible for intelligent document retrieval and re-ranking"""
    
    def __init__(self):
        super().__init__("adaptive_retrieval", "retrieval")
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        
    def chunk_and_embed_files(self, file_paths):
        all_chunks = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            elif ext in [".txt", ".md"]:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(path)
                text = df.to_csv(index=False)
            else:
                continue
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        if all_chunks:
            self.vectorstore = FAISS.from_texts(all_chunks, self.embeddings)

    def update_vectorstore_with_files(self, file_paths):
        self.chunk_and_embed_files(file_paths)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process queries and return relevant documents"""
        self.log_activity("processing_retrieval", {"queries": input_data.get("queries", [])})
        
        if not self.validate_input(input_data, ["queries"]):
            return self.create_response(False, error="Missing required field: queries")
        
        if not self.vectorstore:
            return self.create_response(False, error="Vector store not initialized")
        
        try:
            queries = input_data["queries"]
            original_query = input_data.get("original_query", queries[0] if queries else "")
            
            # Perform multi-query retrieval
            all_documents = []
            for query in queries:
                docs = self._retrieve_documents(query)
                all_documents.extend(docs)
            
            # Remove duplicates and re-rank
            unique_docs = self._remove_duplicates(all_documents)
            re_ranked_docs = self._re_rank_documents(unique_docs, original_query)
            
            # Select top documents
            top_docs = re_ranked_docs[:10]  # Limit to top 10
            
            response_data = {
                "original_query": original_query,
                "retrieved_documents": top_docs,
                "total_queries_processed": len(queries),
                "total_documents_found": len(all_documents),
                "unique_documents": len(unique_docs)
            }
            
            self.log_activity("retrieval_completed", response_data)
            return self.create_response(True, data=response_data)
            
        except Exception as e:
            self.log_activity("error", {"error": str(e)})
            return self.create_response(False, error=f"Error during retrieval: {str(e)}")
    
    def _retrieve_documents(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """Retrieve documents for a single query"""
        try:
            # Vector similarity search
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Convert to standardized format
            formatted_docs = []
            for doc, score in docs:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                })
            
            return formatted_docs
            
        except Exception as e:
            self.log_activity("retrieval_error", {"query": query, "error": str(e)})
            return []
    
    def _remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar documents"""
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            content = doc["content"].strip().lower()
            
            # Check for exact duplicates
            if content in seen_contents:
                continue
            
            # Check for high similarity with existing documents
            is_duplicate = False
            for existing_doc in unique_docs:
                existing_content = existing_doc["content"].strip().lower()
                similarity = SequenceMatcher(None, content, existing_content).ratio()
                
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    # Keep the one with better score
                    if doc["similarity_score"] < existing_doc["similarity_score"]:
                        unique_docs.remove(existing_doc)
                        unique_docs.append(doc)
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.add(content)
        
        return unique_docs
    
    def _re_rank_documents(self, documents: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Re-rank documents based on multiple factors"""
        if not documents:
            return []
        
        # Calculate additional ranking factors
        for doc in documents:
            # Content length factor (prefer medium-length documents)
            content_length = len(doc["content"])
            length_score = 1.0 - abs(content_length - 500) / 1000  # Optimal around 500 chars
            length_score = max(0.1, min(1.0, length_score))
            
            # Keyword density factor
            keyword_density = self._calculate_keyword_density(doc["content"], original_query)
            
            # Source quality factor (prefer certain sources)
            source_score = self._calculate_source_score(doc["source"])
            
            # Combined score (weighted average)
            doc["final_score"] = (
                0.4 * (1.0 - doc["similarity_score"]) +  # Lower is better for similarity
                0.2 * length_score +
                0.2 * keyword_density +
                0.2 * source_score
            )
        
        # Sort by final score (higher is better)
        return sorted(documents, key=lambda x: x["final_score"], reverse=True)
    
    def _calculate_keyword_density(self, content: str, query: str) -> float:
        """Calculate keyword density in content"""
        query_words = set(query.lower().split())
        content_words = content.lower().split()
        
        if not content_words:
            return 0.0
        
        keyword_count = sum(1 for word in content_words if word in query_words)
        return keyword_count / len(content_words)
    
    def _calculate_source_score(self, source: str) -> float:
        """Calculate source quality score"""
        # Define source quality weights
        source_weights = {
            "pdf": 1.0,
            "docx": 0.9,
            "txt": 0.8,
            "md": 0.9,
            "csv": 0.7
        }
        
        # Extract file extension
        if "." in source:
            ext = source.split(".")[-1].lower()
            return source_weights.get(ext, 0.5)
        
        return 0.5  # Default score
    
    def update_vectorstore(self, vectorstore: FAISS):
        """Update the vector store reference"""
        self.vectorstore = vectorstore
        self.log_activity("vectorstore_updated", {"status": "success"}) 