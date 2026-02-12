"""
Ticketing System Integration
Creates and manages tickets in Jira and ServiceNow
"""

import logging
from typing import Dict, Optional
from datetime import datetime

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


class JiraConnector:
    """Create and manage Jira tickets"""
    
    def __init__(self, jira_url: str, api_token: str, project_key: str, 
                 username: str = "api"):
        """
        Initialize Jira connector
        
        Args:
            jira_url: Jira instance URL (e.g., https://jira.company.com)
            api_token: Jira API token
            project_key: Jira project key (e.g., SEC)
            username: Jira username (default: api)
        """
        self.jira_url = jira_url.rstrip("/")
        self.api_token = api_token
        self.project_key = project_key
        self.username = username
        self.auth = HTTPBasicAuth(username, api_token)
    
    def create_ticket(self, 
                     title: str,
                     description: str,
                     issue_type: str = "Task",
                     priority: str = "Medium",
                     labels: Optional[list] = None,
                     custom_fields: Optional[Dict] = None) -> Optional[str]:
        """
        Create a Jira ticket
        
        Args:
            title: Ticket title/summary
            description: Ticket description
            issue_type: Issue type (Task, Bug, Incident, etc.)
            priority: Priority (Lowest, Low, Medium, High, Highest)
            labels: Optional list of labels
            custom_fields: Optional custom field values
            
        Returns:
            Ticket key if successful (e.g., SEC-1234), None otherwise
        """
        try:
            priority_map = {
                "Low": 3,
                "Medium": 2,
                "High": 1,
                "Critical": 0
            }
            
            payload = {
                "fields": {
                    "project": {"key": self.project_key},
                    "summary": title,
                    "description": description,
                    "issuetype": {"name": issue_type},
                    "priority": {"name": priority}
                }
            }
            
            if labels:
                payload["fields"]["labels"] = labels
            
            if custom_fields:
                payload["fields"].update(custom_fields)
            
            response = requests.post(
                f"{self.jira_url}/rest/api/3/issue",
                auth=self.auth,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 201:
                ticket_key = response.json().get("key")
                logger.info(f"Jira ticket created: {ticket_key}")
                return ticket_key
            else:
                logger.error(f"Jira API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create Jira ticket: {e}")
            return None
    
    def update_ticket(self, 
                     ticket_key: str,
                     fields: Dict) -> bool:
        """
        Update a Jira ticket
        
        Args:
            ticket_key: Ticket key (e.g., SEC-1234)
            fields: Fields to update (dict of field_name: value)
            
        Returns:
            True if successful
        """
        try:
            payload = {
                "fields": fields
            }
            
            response = requests.put(
                f"{self.jira_url}/rest/api/3/issue/{ticket_key}",
                auth=self.auth,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"Jira ticket updated: {ticket_key}")
                return True
            else:
                logger.error(f"Jira API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update Jira ticket: {e}")
            return False
    
    def add_comment(self, ticket_key: str, comment: str) -> bool:
        """
        Add a comment to a Jira ticket
        
        Args:
            ticket_key: Ticket key
            comment: Comment text
            
        Returns:
            True if successful
        """
        try:
            payload = {
                "body": {
                    "version": 1,
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": comment}
                            ]
                        }
                    ]
                }
            }
            
            response = requests.post(
                f"{self.jira_url}/rest/api/3/issue/{ticket_key}/comments",
                auth=self.auth,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 201:
                logger.info(f"Comment added to Jira ticket: {ticket_key}")
                return True
            else:
                logger.error(f"Jira API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add Jira comment: {e}")
            return False
    
    def get_ticket(self, ticket_key: str) -> Optional[Dict]:
        """
        Get Jira ticket details
        
        Args:
            ticket_key: Ticket key
            
        Returns:
            Ticket details dict or None
        """
        try:
            response = requests.get(
                f"{self.jira_url}/rest/api/3/issue/{ticket_key}",
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Jira API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get Jira ticket: {e}")
            return None


class ServiceNowConnector:
    """Create and manage ServiceNow incidents"""
    
    def __init__(self, snow_instance: str, api_user: str, api_password: str):
        """
        Initialize ServiceNow connector
        
        Args:
            snow_instance: ServiceNow instance URL
            api_user: API user
            api_password: API password
        """
        self.snow_instance = snow_instance.rstrip("/")
        self.api_user = api_user
        self.api_password = api_password
        self.auth = HTTPBasicAuth(api_user, api_password)
    
    def create_incident(self,
                       short_description: str,
                       description: str,
                       urgency: int = 2,
                       impact: int = 2,
                       assignment_group: Optional[str] = None,
                       custom_fields: Optional[Dict] = None) -> Optional[str]:
        """
        Create a ServiceNow incident
        
        Args:
            short_description: Incident title
            description: Incident description
            urgency: Urgency level (1=High, 2=Medium, 3=Low)
            impact: Impact level (1=High, 2=Medium, 3=Low)
            assignment_group: Assignment group name
            custom_fields: Optional custom fields
            
        Returns:
            Incident number if successful, None otherwise
        """
        try:
            payload = {
                "short_description": short_description,
                "description": description,
                "urgency": urgency,
                "impact": impact
            }
            
            if assignment_group:
                payload["assignment_group"] = assignment_group
            
            if custom_fields:
                payload.update(custom_fields)
            
            response = requests.post(
                f"{self.snow_instance}/api/now/table/incident",
                auth=self.auth,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                incident_number = response.json().get("result", {}).get("number")
                logger.info(f"ServiceNow incident created: {incident_number}")
                return incident_number
            else:
                logger.error(f"ServiceNow API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create ServiceNow incident: {e}")
            return None
    
    def update_incident(self, incident_number: str, fields: Dict) -> bool:
        """
        Update a ServiceNow incident
        
        Args:
            incident_number: Incident number (e.g., INC0010001)
            fields: Fields to update
            
        Returns:
            True if successful
        """
        try:
            response = requests.patch(
                f"{self.snow_instance}/api/now/table/incident",
                auth=self.auth,
                json={
                    "number": incident_number,
                    **fields
                },
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"ServiceNow incident updated: {incident_number}")
                return True
            else:
                logger.error(f"ServiceNow API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update ServiceNow incident: {e}")
            return False
    
    def add_work_note(self, incident_number: str, work_note: str) -> bool:
        """
        Add a work note to a ServiceNow incident
        
        Args:
            incident_number: Incident number
            work_note: Work note text
            
        Returns:
            True if successful
        """
        try:
            response = requests.patch(
                f"{self.snow_instance}/api/now/table/incident",
                auth=self.auth,
                json={
                    "number": incident_number,
                    "work_notes": work_note
                },
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Work note added to ServiceNow incident: {incident_number}")
                return True
            else:
                logger.error(f"ServiceNow API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add ServiceNow work note: {e}")
            return False
    
    def get_incident(self, incident_number: str) -> Optional[Dict]:
        """
        Get ServiceNow incident details
        
        Args:
            incident_number: Incident number
            
        Returns:
            Incident details dict or None
        """
        try:
            response = requests.get(
                f"{self.snow_instance}/api/now/table/incident?number={incident_number}",
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json().get("result", [])
                return result[0] if result else None
            else:
                logger.error(f"ServiceNow API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get ServiceNow incident: {e}")
            return None
