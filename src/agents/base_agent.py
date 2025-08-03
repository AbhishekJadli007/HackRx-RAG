import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    def log_activity(self, activity: str, data: Dict[str, Any] = None):
        """Log agent activity for monitoring and debugging"""
        log_entry = {
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "activity": activity,
            "data": data
        }
        self.logger.info(json.dumps(log_entry))
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate that required fields are present in input data"""
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        return True
    
    def create_response(self, success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
        """Create a standardized response format"""
        response = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "timestamp": time.time(),
            "success": success
        }
        
        if success and data is not None:
            response["data"] = data
        elif not success and error:
            response["error"] = error
            
        return response 