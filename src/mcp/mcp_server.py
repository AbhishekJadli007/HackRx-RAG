import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agents.structured_data_agent import StructuredDataExtractionAgent
from agents.retrieval_agent import AdaptiveRetrievalAgent
from agents.query_reformulation_agent import QueryReformulationAgent
import time

# Load environment variables from the neurofetch-ui directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class MCPMessage:
    id: str
    type: MessageType
    agent_id: str
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

class MCPServer:
    """MCP Server for managing agent communication and LLM interactions"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.agents: Dict[str, Any] = {}
        self.llm_connections: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger("mcp_server")
        
    async def start(self):
        """Start the MCP server"""
        self.logger.info(f"Starting MCP Server on {self.host}:{self.port}")
        
        # Start message processor
        asyncio.create_task(self._process_messages())
        
        # Start server
        server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        
        async with server:
            await server.serve_forever()
    
    async def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Register an agent with the MCP server"""
        self.agents[agent_id] = {
            "type": agent_type,
            "capabilities": capabilities,
            "status": "active",
            "registered_at": datetime.now()
        }
        self.logger.info(f"Agent {agent_id} ({agent_type}) registered")
    
    async def register_llm_connection(self, llm_id: str, llm_type: str, endpoint: str):
        """Register an LLM connection"""
        self.llm_connections[llm_id] = {
            "type": llm_type,
            "endpoint": endpoint,
            "status": "active",
            "registered_at": datetime.now()
        }
        self.logger.info(f"LLM {llm_id} ({llm_type}) registered at {endpoint}")
    
    async def send_message(self, message: MCPMessage):
        """Send a message to the queue for processing"""
        await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._handle_message(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: MCPMessage):
        """Handle individual messages"""
        try:
            if message.type == MessageType.REQUEST:
                await self._handle_request(message)
            elif message.type == MessageType.RESPONSE:
                await self._handle_response(message)
            elif message.type == MessageType.ERROR:
                await self._handle_error(message)
            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
        except Exception as e:
            self.logger.error(f"Error handling message {message.id}: {e}")
    
    async def _handle_request(self, message: MCPMessage):
        """Handle agent requests"""
        agent_id = message.agent_id
        data = message.data
        
        if agent_id not in self.agents:
            await self._send_error_response(message, f"Agent {agent_id} not registered")
            return
        
        # Route request to appropriate LLM
        llm_id = data.get("llm_id", "default")
        if llm_id not in self.llm_connections:
            await self._send_error_response(message, f"LLM {llm_id} not available")
            return
        
        # Process with LLM
        try:
            llm_response = await self._call_llm(llm_id, data)
            response_message = MCPMessage(
                id=str(uuid.uuid4()),
                type=MessageType.RESPONSE,
                agent_id=agent_id,
                data=llm_response,
                timestamp=datetime.now(),
                correlation_id=message.id
            )
            await self.send_message(response_message)
        except Exception as e:
            await self._send_error_response(message, str(e))
    
    async def _handle_response(self, message: MCPMessage):
        """Handle responses from LLMs"""
        self.logger.info(f"Response received for agent {message.agent_id}")
        # Notify waiting agents
        # This would typically involve notifying the specific agent waiting for this response
    
    async def _handle_error(self, message: MCPMessage):
        """Handle error messages"""
        self.logger.error(f"Error from agent {message.agent_id}: {message.data}")
    
    async def _handle_heartbeat(self, message: MCPMessage):
        """Handle heartbeat messages"""
        agent_id = message.agent_id
        if agent_id in self.agents:
            self.agents[agent_id]["last_heartbeat"] = datetime.now()
    
    async def _call_llm(self, llm_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the specified LLM with the given data"""
        llm_info = self.llm_connections[llm_id]
        
        # This is a simplified LLM call - in practice, you'd have proper LLM client connections
        if llm_info["type"] == "ollama":
            return await self._call_ollama(llm_info["endpoint"], data)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_info['type']}")
    
    async def _call_ollama(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama LLM"""
        # Simplified Ollama call - you'd implement proper async HTTP calls here
        prompt = data.get("prompt", "")
        model = data.get("model", "llama3")
        
        # This would be an actual HTTP call to Ollama
        # For now, return a mock response
        return {
            "response": f"Mock response from Ollama ({model}): {prompt[:50]}...",
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _send_error_response(self, original_message: MCPMessage, error: str):
        """Send error response"""
        error_message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            agent_id=original_message.agent_id,
            data={"error": error, "original_request": original_message.data},
            timestamp=datetime.now(),
            correlation_id=original_message.id
        )
        await self.send_message(error_message)
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connections"""
        addr = writer.get_extra_info('peername')
        self.logger.info(f"Client connected from {addr}")
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                
                # Parse message
                message_data = json.loads(data.decode())
                message = MCPMessage(
                    id=message_data["id"],
                    type=MessageType(message_data["type"]),
                    agent_id=message_data["agent_id"],
                    data=message_data["data"],
                    timestamp=datetime.fromisoformat(message_data["timestamp"]),
                    correlation_id=message_data.get("correlation_id")
                )
                
                await self.send_message(message)
                
                # Send acknowledgment
                response = {"status": "received", "message_id": message.id}
                writer.write(json.dumps(response).encode())
                await writer.drain()
                
        except Exception as e:
            self.logger.error(f"Error handling client {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"Client {addr} disconnected")

# Global MCP server instance
mcp_server = MCPServer()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServer")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
structured_agent = StructuredDataExtractionAgent()
retrieval_agent = AdaptiveRetrievalAgent()
query_reformulation_agent = QueryReformulationAgent()

AGENTS = {
    "structured_data_extraction": structured_agent,
    "adaptive_retrieval": retrieval_agent,
    "query_reformulation": query_reformulation_agent,
}

# Health check endpoints for each agent
@app.route("/health/structured_data_extraction")
def health_structured():
    try:
        result = structured_agent.process({"query": "health check"})
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/health/adaptive_retrieval")
def health_retrieval():
    try:
        result = retrieval_agent.process({"queries": ["health check"]})
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/health/query_reformulation")
def health_query_reformulation():
    try:
        result = query_reformulation_agent.process({"query": "health check"})
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/health/llm")
def health_llm():
    try:
        result = llm("health check")
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

# Main orchestration endpoint
@app.route("/route_query", methods=["POST"])
def route_query():
    data = request.json
    query = data.get("query")
    context = data.get("context", {})
    logger.info(f"Received query: {query}")
    start_time = time.time()
    # 1. LLM tries to answer first
    llm_response = llm(query)
    logger.info(f"LLM initial response: {llm_response}")
    if not is_llm_confident(llm_response):
        # 2. LLM not confident, detect intent
        intent = detect_intent(query)
        logger.info(f"LLM not confident. Detected intent: {intent}")
        if intent in ["table", "chat"]:
            agent = structured_agent
            agent_name = "structured_data_extraction"
            agent_input = {"query": query, **context}
        else:
            agent = retrieval_agent
            agent_name = "adaptive_retrieval"
            agent_input = {"queries": [query], "original_query": query, **context}
        try:
            agent_result = agent.process(agent_input)
            logger.info(f"Agent {agent_name} result: {agent_result}")
            if not agent_result.get("success"):
                # 3. Fallback to LLM-with-tools (simulate by re-asking LLM with agent data)
                llm_tool_response = llm(f"Given this data: {agent_result}, answer the query: {query}")
                logger.info(f"LLM-with-tools response: {llm_tool_response}")
                response = {
                    "agent": "llm_with_tools",
                    "response": llm_tool_response,
                    "trace": ["llm", agent_name, "llm_with_tools"]
                }
            else:
                response = {
                    "agent": agent_name,
                    "response": agent_result,
                    "trace": ["llm", agent_name]
                }
        except Exception as e:
            logger.error(f"Error in agent {agent_name}: {e}")
            response = {
                "agent": agent_name,
                "response": f"Agent error: {str(e)}",
                "trace": ["llm", agent_name, "error"]
            }
    else:
        response = {
            "agent": "llm",
            "response": llm_response,
            "trace": ["llm"]
        }
    response["elapsed"] = round(time.time() - start_time, 2)
    logger.info(f"Final response: {response}")
    return jsonify(response)

def is_llm_confident(llm_response):
    # Simple heuristic: if LLM says "I don't know" or similar, it's not confident
    low_conf_phrases = ["i don't know", "cannot answer", "not sure", "no information", "unknown"]
    resp = str(llm_response).strip().lower()
    return not any(phrase in resp for phrase in low_conf_phrases)

def detect_intent(query):
    prompt = (
        f"Classify the intent of this query: '{query}'. "
        "Respond with one of: 'table', 'chat', 'retrieval', 'definition', 'comparison', 'other'."
    )
    intent = llm(prompt).strip().lower()
    return intent

@app.route("/agents", methods=["GET"])
def list_agents():
    # Return available agents and their health
    agent_status = {}
    for name, agent in AGENTS.items():
        try:
            health = app.test_client().get(f"/health/{name}").json
        except Exception as e:
            health = {"status": "error", "error": str(e)}
        agent_status[name] = health
    return jsonify(agent_status)

if __name__ == "__main__":
    app.run(port=8000) 