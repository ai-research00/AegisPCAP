"""
WebSocket Manager for Real-time Dashboard Updates
Handles streaming alerts, incidents, and flow updates to connected clients
"""

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import List, Set, Dict, Any
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class MessageType(str, Enum):
    """WebSocket message types"""
    ALERT = "alert"
    INCIDENT = "incident"
    FLOW_UPDATE = "flow_update"
    STATISTICS = "statistics"
    TOPOLOGY_UPDATE = "topology_update"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: MessageType
    timestamp: datetime
    data: Dict[str, Any]


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """
    Manage WebSocket connections and broadcast updates
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._broadcaster_task = None
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = {"*"}  # Subscribe to all by default
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to AegisPCAP Dashboard",
            "client_count": len(self.active_connections)
        })
    
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe to a specific channel"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(channel)
            logger.debug(f"Client subscribed to {channel}")
    
    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe from a channel"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(channel)
            logger.debug(f"Client unsubscribed from {channel}")
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """
        Broadcast message to all clients subscribed to a channel
        """
        disconnected = []
        
        for connection, subscriptions in self.subscriptions.items():
            if channel in subscriptions or "*" in subscriptions:
                try:
                    await connection.send_json({
                        "type": channel,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": message
                    })
                except Exception as e:
                    logger.warning(f"Error sending message: {str(e)}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "timestamp": datetime.utcnow().isoformat(),
                    **message
                })
            except Exception as e:
                logger.warning(f"Error broadcasting: {str(e)}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connections alive"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                await self.broadcast_to_all({
                    "type": "heartbeat",
                    "connection_count": len(self.active_connections)
                })
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    async def receive_message(self, websocket: WebSocket) -> Dict[str, Any]:
        """Receive and parse message from client"""
        try:
            data = await websocket.receive_text()
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received")
            return None
        except WebSocketDisconnect:
            raise


# ============================================================================
# Global Connection Manager Instance
# ============================================================================

manager = ConnectionManager()


# ============================================================================
# WebSocket Router
# ============================================================================

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time dashboard updates
    
    Subscribable Channels:
        - alerts: Real-time security alerts
        - incidents: Incident updates
        - flows: Flow statistics updates
        - topology: Network topology changes
        - statistics: System statistics
    
    Example Client:
        ```javascript
        const ws = new WebSocket('ws://localhost:8080/api/dashboard/ws');
        
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Update:', message);
        };
        
        // Subscribe to alerts
        ws.send(JSON.stringify({
            type: 'subscribe',
            channel: 'alerts'
        }));
        ```
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            message = await manager.receive_message(websocket)
            
            if message is None:
                continue
            
            message_type = message.get("type")
            
            # Handle subscription
            if message_type == "subscribe":
                channel = message.get("channel", "alerts")
                await manager.subscribe(websocket, channel)
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": channel,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.info(f"Client subscribed to: {channel}")
            
            # Handle unsubscription
            elif message_type == "unsubscribe":
                channel = message.get("channel", "alerts")
                await manager.unsubscribe(websocket, channel)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "channel": channel,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Handle ping (for connection keep-alive)
            elif message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                logger.debug(f"Received message type: {message_type}")
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(websocket)


# ============================================================================
# Alert Broadcasting Functions
# ============================================================================

async def broadcast_alert(alert_data: Dict[str, Any]):
    """
    Broadcast a new alert to all subscribed clients
    
    Args:
        alert_data: Alert information to broadcast
    """
    await manager.broadcast_to_channel("alerts", {
        "alert_id": alert_data.get("id"),
        "severity": alert_data.get("severity"),
        "title": alert_data.get("title"),
        "timestamp": alert_data.get("timestamp", datetime.utcnow()).isoformat(),
        "flow_id": alert_data.get("flow_id")
    })


async def broadcast_incident(incident_data: Dict[str, Any]):
    """
    Broadcast a new incident to all subscribed clients
    
    Args:
        incident_data: Incident information to broadcast
    """
    await manager.broadcast_to_channel("incidents", {
        "incident_id": incident_data.get("id"),
        "title": incident_data.get("title"),
        "severity": incident_data.get("severity"),
        "timestamp": incident_data.get("timestamp", datetime.utcnow()).isoformat(),
        "affected_flows": incident_data.get("flow_count", 0)
    })


async def broadcast_flow_update(flow_data: Dict[str, Any]):
    """
    Broadcast flow statistics update
    
    Args:
        flow_data: Flow information to broadcast
    """
    await manager.broadcast_to_channel("flows", {
        "flow_id": flow_data.get("id"),
        "src_ip": flow_data.get("src_ip"),
        "dst_ip": flow_data.get("dst_ip"),
        "risk_score": flow_data.get("risk_score"),
        "packet_count": flow_data.get("packet_count"),
        "byte_count": flow_data.get("byte_count")
    })


async def broadcast_topology_update(topology_data: Dict[str, Any]):
    """
    Broadcast network topology update
    
    Args:
        topology_data: Topology information to broadcast
    """
    await manager.broadcast_to_channel("topology", topology_data)


async def broadcast_statistics(stats_data: Dict[str, Any]):
    """
    Broadcast system statistics update
    
    Args:
        stats_data: Statistics to broadcast
    """
    await manager.broadcast_to_channel("statistics", {
        "total_flows": stats_data.get("total_flows"),
        "total_alerts": stats_data.get("total_alerts"),
        "total_incidents": stats_data.get("total_incidents"),
        "active_threats": stats_data.get("active_threats"),
        "timestamp": datetime.utcnow().isoformat()
    })


# ============================================================================
# Connection Status Endpoint
# ============================================================================

@router.get("/ws-status")
async def websocket_status():
    """Get WebSocket connection status"""
    return {
        "active_connections": manager.get_connection_count(),
        "timestamp": datetime.utcnow().isoformat(),
        "supported_channels": [
            "alerts",
            "incidents",
            "flows",
            "topology",
            "statistics"
        ]
    }
