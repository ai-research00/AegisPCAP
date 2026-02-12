"""
Phase 6.1: Dashboard API Examples & Usage Guide

This module contains runnable examples for the FastAPI dashboard endpoints.
Each example demonstrates key functionality of the dashboard API.

Installation:
    pip install fastapi uvicorn sqlalchemy httpx websockets

Running the Dashboard:
    python -m src.dashboard.app

API Documentation:
    http://localhost:8080/docs (Swagger UI)
    http://localhost:8080/redoc (ReDoc)
"""

import asyncio
import json
from datetime import datetime, timedelta
import httpx
import websockets
from typing import Dict, List, Any


# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8080"
WS_URL = "ws://localhost:8080/api/dashboard/ws"
API_PREFIX = "/api/dashboard"


# ============================================================================
# Example 1: Health Check & Configuration
# ============================================================================

async def example_health_check():
    """Example 1: Check dashboard health and configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Health Check & Configuration")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Health check
        print("\n1. Health Check:")
        response = await client.get(f"{BASE_URL}{API_PREFIX}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Configuration
        print("\n2. Dashboard Configuration:")
        response = await client.get(f"{BASE_URL}{API_PREFIX}/config")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


# ============================================================================
# Example 2: Get Dashboard Overview
# ============================================================================

async def example_dashboard_overview():
    """Example 2: Get complete dashboard overview with all key metrics"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dashboard Overview")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}{API_PREFIX}/overview")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nStatus: {response.status_code}")
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Status: {data.get('status')}")
            
            if "metrics" in data:
                metrics = data["metrics"]
                print("\nSystem Metrics:")
                if "stats" in metrics:
                    stats = metrics["stats"]
                    print(f"  Total Flows: {stats.get('total_flows', 0)}")
                    print(f"  Total Alerts: {stats.get('total_alerts', 0)}")
                    print(f"  Total Incidents: {stats.get('total_incidents', 0)}")
                    print(f"  High-Risk Flows: {stats.get('high_risk_flows', 0)}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


# ============================================================================
# Example 3: List Flows with Pagination
# ============================================================================

async def example_list_flows():
    """Example 3: List flows with pagination, filtering, and sorting"""
    print("\n" + "="*70)
    print("EXAMPLE 3: List Flows with Pagination & Filtering")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Example 3a: Get first page of flows
        print("\n3a. First page of flows (default pagination):")
        params = {
            "page": 1,
            "page_size": 10,
            "sort_by": "timestamp",
            "sort_order": "desc"
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/flows", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {response.status_code}")
            print(f"Total Flows: {data['pagination'].get('total', 0)}")
            print(f"Pages: {data['pagination'].get('pages', 0)}")
            
            if data["data"]:
                print(f"\nFirst Flow:")
                flow = data["data"][0]
                print(f"  Flow ID: {flow.get('flow_id')}")
                print(f"  Source: {flow.get('src_ip')}:{flow.get('src_port')}")
                print(f"  Destination: {flow.get('dst_ip')}:{flow.get('dst_port')}")
                print(f"  Protocol: {flow.get('protocol')}")
                print(f"  Risk Score: {flow.get('risk_score')}")
        
        # Example 3b: Filter by protocol
        print("\n3b. Filter flows by protocol (TCP):")
        params = {
            "protocol": "TCP",
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/flows", params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"TCP Flows Found: {data['pagination'].get('total', 0)}")
        
        # Example 3c: Filter by risk score
        print("\n3c. High-risk flows (score >= 70):")
        params = {
            "min_risk_score": 70,
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/flows", params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"High-Risk Flows: {data['pagination'].get('total', 0)}")


# ============================================================================
# Example 4: Get Flow Details
# ============================================================================

async def example_flow_detail():
    """Example 4: Get detailed information about a specific flow"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Flow Details")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # First, get a flow ID
        response = await client.get(f"{BASE_URL}{API_PREFIX}/flows", params={"page": 1, "page_size": 1})
        
        if response.status_code == 200 and response.json()["data"]:
            flow_id = response.json()["data"][0]["flow_id"]
            
            print(f"\nGetting details for Flow ID: {flow_id}")
            response = await client.get(f"{BASE_URL}{API_PREFIX}/flows/{flow_id}")
            
            if response.status_code == 200:
                flow = response.json()
                print(f"Status: {response.status_code}")
                print(f"Flow Details:")
                print(f"  ID: {flow.get('flow_id')}")
                print(f"  Source: {flow.get('src_ip')}:{flow.get('src_port')}")
                print(f"  Destination: {flow.get('dst_ip')}:{flow.get('dst_port')}")
                print(f"  Protocol: {flow.get('protocol')}")
                print(f"  Duration: {flow.get('duration_seconds')} seconds")
                print(f"  Packets: {flow.get('packets')}")
                print(f"  Bytes: {flow.get('bytes')}")
                print(f"  Risk Score: {flow.get('risk_score')}")
                print(f"  Associated Alerts: {len(flow.get('alerts', []))}")


# ============================================================================
# Example 5: List Alerts with Filtering
# ============================================================================

async def example_list_alerts():
    """Example 5: List security alerts with severity filtering"""
    print("\n" + "="*70)
    print("EXAMPLE 5: List Alerts with Filtering")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Example 5a: All alerts
        print("\n5a. All alerts (paginated):")
        params = {
            "page": 1,
            "page_size": 10,
            "sort_by": "timestamp",
            "sort_order": "desc"
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/alerts", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Alerts: {data['pagination'].get('total', 0)}")
        
        # Example 5b: High-severity alerts only
        print("\n5b. High-severity alerts only:")
        params = {
            "severity": "high",
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/alerts", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"High-Severity Alerts: {data['pagination'].get('total', 0)}")
        
        # Example 5c: Unacknowledged alerts
        print("\n5c. Unacknowledged alerts:")
        params = {
            "acknowledged": False,
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/alerts", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Unacknowledged Alerts: {data['pagination'].get('total', 0)}")


# ============================================================================
# Example 6: Get Alert Details
# ============================================================================

async def example_alert_detail():
    """Example 6: Get detailed information about a specific alert"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Alert Details")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Get first alert
        response = await client.get(f"{BASE_URL}{API_PREFIX}/alerts", params={"page": 1, "page_size": 1})
        
        if response.status_code == 200 and response.json()["data"]:
            alert_id = response.json()["data"][0]["alert_id"]
            
            print(f"\nGetting details for Alert ID: {alert_id}")
            response = await client.get(f"{BASE_URL}{API_PREFIX}/alerts/{alert_id}")
            
            if response.status_code == 200:
                alert = response.json()
                print(f"Alert Details:")
                print(f"  ID: {alert.get('alert_id')}")
                print(f"  Severity: {alert.get('severity')}")
                print(f"  Title: {alert.get('title')}")
                print(f"  Detector: {alert.get('detector')}")
                print(f"  Confidence: {alert.get('confidence')}")
                print(f"  Flow ID: {alert.get('flow_id')}")
                print(f"  Acknowledged: {alert.get('acknowledged')}")


# ============================================================================
# Example 7: Acknowledge Alert
# ============================================================================

async def example_acknowledge_alert():
    """Example 7: Acknowledge a security alert"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Acknowledge Alert")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Get first unacknowledged alert
        response = await client.get(
            f"{BASE_URL}{API_PREFIX}/alerts",
            params={"acknowledged": False, "page": 1, "page_size": 1}
        )
        
        if response.status_code == 200 and response.json()["data"]:
            alert_id = response.json()["data"][0]["alert_id"]
            
            print(f"\nAcknowledging Alert ID: {alert_id}")
            response = await client.post(f"{BASE_URL}{API_PREFIX}/alerts/{alert_id}/acknowledge")
            
            if response.status_code == 200:
                print(f"Response: {json.dumps(response.json(), indent=2)}")


# ============================================================================
# Example 8: List Incidents
# ============================================================================

async def example_list_incidents():
    """Example 8: List security incidents"""
    print("\n" + "="*70)
    print("EXAMPLE 8: List Incidents")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Example 8a: Open incidents
        print("\n8a. Open incidents:")
        params = {
            "status": "open",
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/incidents", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Open Incidents: {data['pagination'].get('total', 0)}")
        
        # Example 8b: High-severity incidents
        print("\n8b. High-severity incidents:")
        params = {
            "severity": "high",
            "page": 1,
            "page_size": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/incidents", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"High-Severity Incidents: {data['pagination'].get('total', 0)}")


# ============================================================================
# Example 9: Analytics - Threat Timeline
# ============================================================================

async def example_threat_timeline():
    """Example 9: Get threat activity timeline"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Threat Timeline Analytics")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Get last 24 hours of threat activity
        print("\nThreat activity over last 24 hours:")
        params = {"hours": 24}
        response = await client.get(f"{BASE_URL}{API_PREFIX}/analytics/threat-timeline", params=params)
        
        if response.status_code == 200:
            timeline = response.json()
            print(f"Timeline Title: {timeline.get('title')}")
            print(f"Data Points: {len(timeline.get('data_points', []))}")


# ============================================================================
# Example 10: Analytics - Top Attacking IPs
# ============================================================================

async def example_top_ips():
    """Example 10: Get top attacking IP addresses"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Top Attacking IPs")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Top source IPs
        print("\nTop 10 source IPs by alert count:")
        params = {
            "direction": "src",
            "limit": 10
        }
        response = await client.get(f"{BASE_URL}{API_PREFIX}/analytics/top-ips", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Results: {len(data.get('items', []))}")
            for item in data.get('items', [])[:3]:
                print(f"  #{item.get('rank')}: {item.get('label')} ({item.get('value')} alerts)")


# ============================================================================
# Example 11: System Statistics
# ============================================================================

async def example_statistics():
    """Example 11: Get system-wide statistics"""
    print("\n" + "="*70)
    print("EXAMPLE 11: System Statistics")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}{API_PREFIX}/analytics/statistics")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"System Statistics:")
            print(f"  Total Flows: {stats.get('total_flows', 0)}")
            print(f"  Total Alerts: {stats.get('total_alerts', 0)}")
            print(f"  Total Incidents: {stats.get('total_incidents', 0)}")
            print(f"  Active Incidents: {stats.get('active_incidents', 0)}")
            print(f"  Average Risk Score: {stats.get('average_risk_score', 0):.2f}")
            print(f"  Unique Source IPs: {stats.get('unique_source_ips', 0)}")


# ============================================================================
# Example 12: WebSocket Real-time Updates
# ============================================================================

async def example_websocket():
    """Example 12: Connect to WebSocket for real-time updates"""
    print("\n" + "="*70)
    print("EXAMPLE 12: WebSocket Real-time Updates")
    print("="*70)
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"\nConnected to WebSocket: {WS_URL}")
            
            # Subscribe to alerts
            print("\nSubscribing to alerts channel...")
            subscribe_msg = {
                "type": "subscribe",
                "channel": "alerts"
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Receive initial confirmation
            confirmation = await websocket.recv()
            print(f"Confirmation: {confirmation}")
            
            # Listen for updates (timeout after 10 seconds for demo)
            print("\nListening for real-time alerts (10 second timeout)...")
            try:
                start_time = datetime.now()
                while (datetime.now() - start_time).total_seconds() < 10:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2)
                    data = json.loads(message)
                    if data.get("type") != "heartbeat":
                        print(f"Received: {json.dumps(data, indent=2)}")
            except asyncio.TimeoutError:
                print("(No messages received - this is normal if no alerts are triggered)")
    
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")
        print("(Make sure the dashboard is running)")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("AEGISPCAP DASHBOARD API - EXAMPLES")
    print("="*70)
    print(f"\nBase URL: {BASE_URL}")
    print(f"API Documentation: {BASE_URL}/docs")
    
    try:
        # Run examples
        await example_health_check()
        await example_dashboard_overview()
        await example_list_flows()
        await example_flow_detail()
        await example_list_alerts()
        await example_alert_detail()
        await example_acknowledge_alert()
        await example_list_incidents()
        await example_threat_timeline()
        await example_top_ips()
        await example_statistics()
        await example_websocket()
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("\nMake sure the dashboard API is running:")
        print(f"  python -m src.dashboard.app")


if __name__ == "__main__":
    asyncio.run(main())
