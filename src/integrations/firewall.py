"""
Firewall & Network Control Integration
Automatic IP blocking and DNS sinkholing
"""

import logging
from typing import Optional, List
from abc import ABC, abstractmethod

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


class FirewallConnector(ABC):
    """Abstract base class for firewall connectors"""
    
    @abstractmethod
    def block_ip(self, ip: str, reason: str = "Threat Detection") -> bool:
        """Block an IP address"""
        pass
    
    @abstractmethod
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address"""
        pass
    
    @abstractmethod
    def is_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked"""
        pass


class PFSenseConnector(FirewallConnector):
    """pfSense firewall integration"""
    
    def __init__(self, host: str, api_key: str, api_secret: str, verify_ssl: bool = True):
        """
        Initialize pfSense connector
        
        Args:
            host: pfSense host URL (e.g., https://192.168.1.1:8443)
            api_key: API key
            api_secret: API secret
            verify_ssl: Verify SSL certificate
        """
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.verify_ssl = verify_ssl
    
    def block_ip(self, ip: str, reason: str = "Threat Detection") -> bool:
        """
        Block an IP on pfSense
        
        Args:
            ip: IP address to block
            reason: Reason for blocking
            
        Returns:
            True if successful
        """
        try:
            # Add to pfSense deny list
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "ip": ip,
                "alias": "AegisPCAP_Blocked",
                "description": reason
            }
            
            response = requests.post(
                f"{self.host}/api/firewall/alias_entry/create",
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"pfSense: Blocked IP {ip}")
                return True
            else:
                logger.error(f"pfSense API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to block IP on pfSense: {e}")
            return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP on pfSense"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.delete(
                f"{self.host}/api/firewall/alias_entry/delete/{ip}",
                headers=headers,
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"pfSense: Unblocked IP {ip}")
                return True
            else:
                logger.error(f"pfSense API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unblock IP on pfSense: {e}")
            return False
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked on pfSense"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.get(
                f"{self.host}/api/firewall/alias_entry/get/{ip}",
                headers=headers,
                verify=self.verify_ssl,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to check if IP is blocked on pfSense: {e}")
            return False


class FortinetConnector(FirewallConnector):
    """Fortinet FortiGate firewall integration"""
    
    def __init__(self, host: str, api_token: str, verify_ssl: bool = True):
        """
        Initialize Fortinet connector
        
        Args:
            host: FortiGate host URL
            api_token: API token
            verify_ssl: Verify SSL certificate
        """
        self.host = host.rstrip("/")
        self.api_token = api_token
        self.verify_ssl = verify_ssl
    
    def block_ip(self, ip: str, reason: str = "Threat Detection") -> bool:
        """Block an IP on FortiGate"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            payload = {
                "name": f"AegisPCAP_{ip}",
                "address": ip,
                "description": reason,
                "type": "ipmask",
                "subnet": f"{ip}/32"
            }
            
            response = requests.post(
                f"{self.host}/api/v2/cmdb/firewall/address",
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"FortiGate: Blocked IP {ip}")
                return True
            else:
                logger.error(f"FortiGate API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to block IP on FortiGate: {e}")
            return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP on FortiGate"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            response = requests.delete(
                f"{self.host}/api/v2/cmdb/firewall/address/AegisPCAP_{ip}",
                headers=headers,
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"FortiGate: Unblocked IP {ip}")
                return True
            else:
                logger.error(f"FortiGate API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unblock IP on FortiGate: {e}")
            return False
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked on FortiGate"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            response = requests.get(
                f"{self.host}/api/v2/cmdb/firewall/address/AegisPCAP_{ip}",
                headers=headers,
                verify=self.verify_ssl,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to check if IP is blocked on FortiGate: {e}")
            return False


class CheckPointConnector(FirewallConnector):
    """Check Point firewall integration"""
    
    def __init__(self, host: str, api_token: str, verify_ssl: bool = True):
        """
        Initialize Check Point connector
        
        Args:
            host: Check Point management server URL
            api_token: API token
            verify_ssl: Verify SSL certificate
        """
        self.host = host.rstrip("/")
        self.api_token = api_token
        self.verify_ssl = verify_ssl
    
    def block_ip(self, ip: str, reason: str = "Threat Detection") -> bool:
        """Block an IP on Check Point"""
        try:
            headers = {
                "X-chkp-sid": self.api_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "name": f"AegisPCAP_{ip}",
                "ipv4-address": ip,
                "comments": reason
            }
            
            response = requests.post(
                f"{self.host}/web_api/add-host",
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Check Point: Blocked IP {ip}")
                return True
            else:
                logger.error(f"Check Point API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to block IP on Check Point: {e}")
            return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP on Check Point"""
        try:
            headers = {
                "X-chkp-sid": self.api_token,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.host}/web_api/delete-object",
                headers=headers,
                json={"name": f"AegisPCAP_{ip}"},
                verify=self.verify_ssl,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Check Point: Unblocked IP {ip}")
                return True
            else:
                logger.error(f"Check Point API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unblock IP on Check Point: {e}")
            return False
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked on Check Point"""
        try:
            headers = {
                "X-chkp-sid": self.api_token,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.host}/web_api/show-object",
                headers=headers,
                json={"name": f"AegisPCAP_{ip}"},
                verify=self.verify_ssl,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to check if IP is blocked on Check Point: {e}")
            return False
