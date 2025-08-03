import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import uuid
from .mcp_server import MCPMessage, MessageType

class MCPClient:
    """MCP Client for agents to communicate with the MCP server"""
    
    def __init__(self, agent_id: str, server_host: str = "localhost", server_port: int = 8000):
        self.agent_id = agent_id
        self.server_host = server_host
        self.server_port = server_port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.logger = logging.getLogger(f"mcp_client.{agent_id}")
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
    async def connect(self):
        """Connect to the MCP server"""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.server_host, self.server_port
            )
            self.connected = True
            self.logger.info(f"Connected to MCP server at {self.server_host}:{self.server_port}")
            
            # Start message listener
            asyncio.create_task(self._listen_for_messages())
            
            # Send registration message
            await self._register_agent()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        self.logger.info("Disconnected from MCP server")
    
    async def _register_agent(self):
        """Register this agent with the MCP server"""
        registration_message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            agent_id=self.agent_id,
            data={
                "action": "register",
                "agent_type": "general",
                "capabilities": ["llm_query", "data_processing"]
            },
            timestamp=datetime.now()
        )
        await self._send_message(registration_message)
    
    async def _send_message(self, message: MCPMessage):
        """Send a message to the MCP server"""
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")
        
        message_data = {
            "id": message.id,
            "type": message.type.value,
            "agent_id": message.agent_id,
            "data": message.data,
            "timestamp": message.timestamp.isoformat(),
            "correlation_id": message.correlation_id
        }
        
        self.writer.write(json.dumps(message_data).encode())
        await self.writer.drain()
    
    async def _listen_for_messages(self):
        """Listen for messages from the MCP server"""
        try:
            while self.connected:
                data = await self.reader.read(1024)
                if not data:
                    break
                
                message_data = json.loads(data.decode())
                message = MCPMessage(
                    id=message_data["id"],
                    type=MessageType(message_data["type"]),
                    agent_id=message_data["agent_id"],
                    data=message_data["data"],
                    timestamp=datetime.fromisoformat(message_data["timestamp"]),
                    correlation_id=message_data.get("correlation_id")
                )
                
                await self._handle_message(message)
                
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
        finally:
            self.connected = False
    
    async def _handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        try:
            if message.type == MessageType.RESPONSE:
                await self._handle_response(message)
            elif message.type == MessageType.ERROR:
                await self._handle_error(message)
            else:
                # Call custom handler if registered
                handler = self.message_handlers.get(message.type)
                if handler:
                    await handler(message)
                else:
                    self.logger.warning(f"No handler for message type: {message.type}")
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _handle_response(self, message: MCPMessage):
        """Handle response messages"""
        correlation_id = message.correlation_id
        if correlation_id in self.pending_requests:
            future = self.pending_requests.pop(correlation_id)
            future.set_result(message.data)
        else:
            self.logger.warning(f"Received response for unknown request: {correlation_id}")
    
    async def _handle_error(self, message: MCPMessage):
        """Handle error messages"""
        correlation_id = message.correlation_id
        if correlation_id in self.pending_requests:
            future = self.pending_requests.pop(correlation_id)
            future.set_exception(Exception(message.data.get("error", "Unknown error")))
        else:
            self.logger.error(f"Error from MCP server: {message.data}")
    
    async def call_llm(self, prompt: str, model: str = "llama3", **kwargs) -> Dict[str, Any]:
        """Call an LLM through the MCP server"""
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        message = MCPMessage(
            id=request_id,
            type=MessageType.REQUEST,
            agent_id=self.agent_id,
            data={
                "action": "llm_call",
                "llm_id": "default",
                "prompt": prompt,
                "model": model,
                **kwargs
            },
            timestamp=datetime.now()
        )
        
        await self._send_message(message)
        
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise TimeoutError("LLM call timed out")
    
    async def send_heartbeat(self):
        """Send heartbeat to the MCP server"""
        if not self.connected:
            return
        
        message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MessageType.HEARTBEAT,
            agent_id=self.agent_id,
            data={"timestamp": datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        await self._send_message(message)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[message_type] = handler
    
    async def start_heartbeat(self, interval: float = 30.0):
        """Start sending periodic heartbeats"""
        while self.connected:
            try:
                await self.send_heartbeat()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                break

class MCPAgentMixin:
    """Mixin to add MCP client capabilities to agents"""
    
    def __init__(self, agent_id: str, mcp_host: str = "localhost", mcp_port: int = 8000):
        self.mcp_client = MCPClient(agent_id, mcp_host, mcp_port)
        self.agent_id = agent_id
    
    async def connect_to_mcp(self):
        """Connect to MCP server"""
        await self.mcp_client.connect()
        # Start heartbeat
        asyncio.create_task(self.mcp_client.start_heartbeat())
    
    async def disconnect_from_mcp(self):
        """Disconnect from MCP server"""
        await self.mcp_client.disconnect()
    
    async def call_llm(self, prompt: str, model: str = "llama3", **kwargs) -> Dict[str, Any]:
        """Call LLM through MCP"""
        return await self.mcp_client.call_llm(prompt, model, **kwargs)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register custom message handler"""
        self.mcp_client.register_message_handler(message_type, handler) 