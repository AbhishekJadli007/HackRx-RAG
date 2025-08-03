from typing import Dict, Any, List
from .base_agent import BaseAgent
import re

class QueryReformulationAgent(BaseAgent):
    """Agent responsible for reformulating and expanding user queries"""
    
    def __init__(self):
        super().__init__("query_reformulation", "query_reformulation")
        
        # Common query patterns and their expansions
        self.query_patterns = {
            "what_is": r"what\s+is\s+(.+)",
            "how_to": r"how\s+to\s+(.+)",
            "compare": r"compare\s+(.+)\s+and\s+(.+)",
            "explain": r"explain\s+(.+)",
            "find": r"find\s+(.+)",
            "list": r"list\s+(.+)"
        }
        
        # Synonyms and related terms for common concepts
        self.synonym_mappings = {
            "capital": ["main city", "headquarters", "center", "hub"],
            "financial": ["monetary", "economic", "fiscal", "budget"],
            "report": ["document", "analysis", "study", "assessment"],
            "data": ["information", "facts", "statistics", "figures"],
            "process": ["procedure", "method", "approach", "technique"],
            "system": ["framework", "platform", "infrastructure", "architecture"]
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query and return reformulated queries"""
        self.log_activity("processing_query", {"original_query": input_data.get("query", "")})
        
        if not self.validate_input(input_data, ["query"]):
            return self.create_response(False, error="Missing required field: query")
        
        original_query = input_data["query"].strip()
        if not original_query:
            return self.create_response(False, error="Empty query provided")
        
        try:
            # Analyze query intent
            intent = self._analyze_intent(original_query)
            
            # Generate reformulated queries
            reformulated_queries = self._reformulate_query(original_query, intent)
            
            # Create response data
            response_data = {
                "original_query": original_query,
                "intent": intent,
                "reformulated_queries": reformulated_queries,
                "primary_query": reformulated_queries[0] if reformulated_queries else original_query
            }
            
            self.log_activity("query_reformulated", response_data)
            return self.create_response(True, data=response_data)
            
        except Exception as e:
            self.log_activity("error", {"error": str(e)})
            return self.create_response(False, error=f"Error reformulating query: {str(e)}")
    
    def _analyze_intent(self, query: str) -> str:
        """Analyze the intent of the query"""
        query_lower = query.lower()
        
        if re.search(self.query_patterns["what_is"], query_lower):
            return "definition"
        elif re.search(self.query_patterns["how_to"], query_lower):
            return "procedure"
        elif re.search(self.query_patterns["compare"], query_lower):
            return "comparison"
        elif re.search(self.query_patterns["explain"], query_lower):
            return "explanation"
        elif re.search(self.query_patterns["find"], query_lower):
            return "search"
        elif re.search(self.query_patterns["list"], query_lower):
            return "enumeration"
        else:
            return "general"
    
    def _reformulate_query(self, query: str, intent: str) -> List[str]:
        """Generate reformulated versions of the query"""
        reformulated = [query]  # Always include original query
        
        # Add synonyms for key terms
        expanded_query = self._expand_with_synonyms(query)
        if expanded_query != query:
            reformulated.append(expanded_query)
        
        # Add intent-specific reformulations
        if intent == "definition":
            reformulated.extend([
                f"define {query}",
                f"what does {query} mean",
                f"explanation of {query}"
            ])
        elif intent == "procedure":
            reformulated.extend([
                f"steps to {query}",
                f"process for {query}",
                f"method to {query}"
            ])
        elif intent == "comparison":
            reformulated.extend([
                f"differences between {query}",
                f"compare {query}",
                f"{query} vs"
            ])
        
        # Add keyword-based reformulations
        keywords = self._extract_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query != query:
                reformulated.append(keyword_query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_reformulated = []
        for q in reformulated:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_reformulated.append(q)
        
        return unique_reformulated[:5]  # Limit to 5 reformulations
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms"""
        expanded = query
        for term, synonyms in self.synonym_mappings.items():
            if term in query.lower():
                # Replace with first synonym
                expanded = re.sub(rf'\b{term}\b', synonyms[0], expanded, flags=re.IGNORECASE)
        return expanded
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Simple keyword extraction - remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords 