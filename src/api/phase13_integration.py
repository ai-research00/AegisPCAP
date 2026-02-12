"""
Phase 13 Integration - Unified Compliance & Privacy API

Orchestrates all 6 compliance modules (GDPR, CCPA, HIPAA, Data Retention,
Audit Logging, Anonymization) through a single unified interface.

Author: AegisPCAP Compliance Team
Date: February 5, 2026
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from compliance.gdpr import GDPRController
from compliance.ccpa import CCPAController
from compliance.hipaa import HIPAAController
from compliance.data_retention import DataRetentionController
from compliance.audit_logging import AuditLoggingController
from compliance.anonymization import AnonymizationController

logger = logging.getLogger(__name__)


@dataclass
class Phase13Configuration:
    """Configuration for Phase 13 compliance system."""
    enable_gdpr: bool = True
    enable_ccpa: bool = True
    enable_hipaa: bool = True
    enable_data_retention: bool = True
    enable_audit_logging: bool = True
    enable_anonymization: bool = True

    # Regulatory settings
    default_retention_days: int = 90
    enforce_minimum_necessary: bool = True
    require_consent_audit: bool = True
    enable_breach_notification: bool = True

    # Privacy settings
    default_k_anonymity: int = 5
    default_epsilon: float = 1.0
    enable_differential_privacy: bool = True

    # Data Protection settings
    encrypt_phi_at_rest: bool = True
    encrypt_phi_in_transit: bool = True
    require_baa_for_processors: bool = True


class Phase13Integration:
    """Unified compliance and privacy orchestration."""

    def __init__(self, config: Optional[Phase13Configuration] = None):
        """
        Initialize Phase 13 integration.

        Args:
            config: Configuration object
        """
        self.config = config or Phase13Configuration()

        # Initialize all 6 compliance modules
        if self.config.enable_gdpr:
            self.gdpr = GDPRController()
        else:
            self.gdpr = None

        if self.config.enable_ccpa:
            self.ccpa = CCPAController()
        else:
            self.ccpa = None

        if self.config.enable_hipaa:
            self.hipaa = HIPAAController()
        else:
            self.hipaa = None

        if self.config.enable_data_retention:
            self.data_retention = DataRetentionController()
        else:
            self.data_retention = None

        if self.config.enable_audit_logging:
            self.audit_logging = AuditLoggingController()
        else:
            self.audit_logging = None

        if self.config.enable_anonymization:
            self.anonymization = AnonymizationController()
        else:
            self.anonymization = None

        self.history: List[Dict] = []
        logger.info(
            "Phase13Integration initialized with config: "
            "GDPR=%s, CCPA=%s, HIPAA=%s, DataRetention=%s, AuditLogging=%s, Anonymization=%s",
            self.config.enable_gdpr,
            self.config.enable_ccpa,
            self.config.enable_hipaa,
            self.config.enable_data_retention,
            self.config.enable_audit_logging,
            self.config.enable_anonymization,
        )

    # ==================== GDPR Operations ====================

    def record_user_consent(
        self,
        user_id: str,
        purposes: List[str],
        version: str = "1.0",
    ) -> Dict:
        """
        Record user consent for processing purposes.

        Args:
            user_id: User ID
            purposes: List of purposes (e.g., ["marketing", "analytics"])
            version: Consent form version

        Returns:
            Consent recording result
        """
        if not self.gdpr:
            return {"status": "disabled", "reason": "GDPR module disabled"}

        result = self.gdpr.record_consent(user_id, purposes, version)

        if self.audit_logging:
            self.audit_logging.log_event(
                "consent_given",
                "system",
                user_id,
                f"Consent recorded for: {', '.join(purposes)}",
            )

        self._log_operation("record_user_consent", result)
        return result

    def initiate_right_to_be_forgotten(self, user_id: str) -> Dict:
        """
        Initiate GDPR right to be forgotten (Article 17).

        Args:
            user_id: User requesting deletion

        Returns:
            Deletion request details
        """
        if not self.gdpr:
            return {"status": "disabled", "reason": "GDPR module disabled"}

        result = self.gdpr.initiate_right_to_be_forgotten(user_id)

        if self.audit_logging:
            self.audit_logging.log_event(
                "data_deletion",
                user_id,
                "all_personal_data",
                "Right to be forgotten initiated",
            )

        self._log_operation("initiate_right_to_be_forgotten", result)
        return result

    # ==================== CCPA Operations ====================

    def handle_consumer_request(
        self,
        request_type: str,
        consumer_id: str,
        email: str,
    ) -> Dict:
        """
        Handle CCPA consumer request (know, delete, opt-out, correct).

        Args:
            request_type: Type of request
            consumer_id: Consumer ID
            email: Consumer email

        Returns:
            Request handling result
        """
        if not self.ccpa:
            return {"status": "disabled", "reason": "CCPA module disabled"}

        result = self.ccpa.handle_consumer_request(
            request_type, consumer_id, email
        )

        if self.audit_logging:
            self.audit_logging.log_event(
                "data_access",
                "system",
                consumer_id,
                f"CCPA {request_type} request received",
            )

        self._log_operation("handle_consumer_request", result)
        return result

    def verify_ccpa_deletion(
        self,
        request_id: str,
        verification_method: str,
        verification_data: Dict,
    ) -> Dict:
        """
        Verify CCPA deletion request.

        Args:
            request_id: Request ID
            verification_method: Verification method
            verification_data: Verification data

        Returns:
            Verification result
        """
        if not self.ccpa:
            return {"status": "disabled", "reason": "CCPA module disabled"}

        result = self.ccpa.verify_deletion_request(
            request_id, verification_method, verification_data
        )

        self._log_operation("verify_ccpa_deletion", result)
        return result

    def track_consumer_opt_out(self, consumer_id: str) -> Dict:
        """
        Track consumer opt-out (sale/sharing).

        Args:
            consumer_id: Consumer ID

        Returns:
            Opt-out tracking result
        """
        if not self.ccpa:
            return {"status": "disabled", "reason": "CCPA module disabled"}

        result = self.ccpa.track_opt_out(consumer_id)

        if self.audit_logging:
            self.audit_logging.log_event(
                "data_export",
                "system",
                consumer_id,
                "Opt-out request recorded",
            )

        self._log_operation("track_consumer_opt_out", result)
        return result

    # ==================== HIPAA Operations ====================

    def register_business_associate(
        self,
        name: str,
        processor_type: str,
        expiry_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Register HIPAA Business Associate Agreement.

        Args:
            name: BA name
            processor_type: Type of processing
            expiry_date: BAA expiry date

        Returns:
            Registration result
        """
        if not self.hipaa:
            return {"status": "disabled", "reason": "HIPAA module disabled"}

        if not expiry_date:
            expiry_date = datetime.utcnow() + timedelta(days=365 * 3)

        result = self.hipaa.register_business_associate(
            name, processor_type, expiry_date
        )

        if self.audit_logging:
            self.audit_logging.log_event(
                "policy_change",
                "system",
                name,
                "Business Associate registered",
            )

        self._log_operation("register_business_associate", result)
        return result

    def classify_protected_health_info(
        self, data_type: str, phi_category: str
    ) -> Dict:
        """
        Classify data as Protected Health Information.

        Args:
            data_type: Type of data
            phi_category: PHI category

        Returns:
            Classification result
        """
        if not self.hipaa:
            return {"status": "disabled", "reason": "HIPAA module disabled"}

        result = self.hipaa.classify_phi(data_type, phi_category)
        self._log_operation("classify_protected_health_info", result)
        return result

    def log_phi_access(
        self,
        user_id: str,
        data_accessed: str,
        purpose: str,
    ) -> Dict:
        """
        Log Protected Health Information access.

        Args:
            user_id: User accessing PHI
            data_accessed: Data being accessed
            purpose: Purpose of access

        Returns:
            Logging result
        """
        if not self.hipaa:
            return {"status": "disabled", "reason": "HIPAA module disabled"}

        result = self.hipaa.log_phi_access(user_id, data_accessed, purpose)

        if self.audit_logging:
            self.audit_logging.log_event(
                "phi_access",
                user_id,
                data_accessed,
                f"PHI accessed for: {purpose}",
            )

        self._log_operation("log_phi_access", result)
        return result

    # ==================== Data Retention Operations ====================

    def create_retention_policy(
        self,
        data_type: str,
        retention_days: int,
        archive_days: Optional[int] = None,
    ) -> Dict:
        """
        Create data retention policy.

        Args:
            data_type: Type of data
            retention_days: Days to retain
            archive_days: Days before archiving

        Returns:
            Policy creation result
        """
        if not self.data_retention:
            return {"status": "disabled", "reason": "DataRetention module disabled"}

        result = self.data_retention.schedule_policy(
            data_type, retention_days, archive_days
        )

        if self.audit_logging:
            self.audit_logging.log_event(
                "policy_change",
                "system",
                data_type,
                f"Retention policy created: {retention_days} days",
            )

        self._log_operation("create_retention_policy", result)
        return result

    def schedule_data_deletion(
        self,
        record_ids: List[str],
        delete_date: datetime,
    ) -> Dict:
        """
        Schedule data for automated deletion.

        Args:
            record_ids: Records to delete
            delete_date: When to delete

        Returns:
            Scheduling result
        """
        if not self.data_retention:
            return {"status": "disabled", "reason": "DataRetention module disabled"}

        schedule_id = self.data_retention.deletion_scheduler.schedule_deletion(
            record_ids, delete_date
        )

        if self.audit_logging:
            self.audit_logging.log_event(
                "data_deletion",
                "system",
                f"{len(record_ids)} records",
                f"Deletion scheduled for {delete_date}",
            )

        return {
            "status": "scheduled",
            "schedule_id": schedule_id,
            "records": len(record_ids),
            "delete_date": delete_date.isoformat(),
        }

    # ==================== Audit Logging Operations ====================

    def log_compliance_event(
        self,
        event_type: str,
        user_id: str,
        subject: str,
        action: str,
        ip_address: Optional[str] = None,
    ) -> Dict:
        """
        Log compliance event to immutable audit trail.

        Args:
            event_type: Type of event
            user_id: User performing action
            subject: What was affected
            action: What was done
            ip_address: User's IP address

        Returns:
            Logging result
        """
        if not self.audit_logging:
            return {"status": "disabled", "reason": "AuditLogging module disabled"}

        result = self.audit_logging.log_event(
            event_type, user_id, subject, action, ip_address
        )

        self._log_operation("log_compliance_event", result)
        return result

    def verify_audit_integrity(self) -> Dict:
        """
        Verify audit trail has not been tampered with.

        Returns:
            Integrity verification result
        """
        if not self.audit_logging:
            return {"status": "disabled", "reason": "AuditLogging module disabled"}

        result = self.audit_logging.verify_audit_trail_integrity()
        self._log_operation("verify_audit_integrity", result)
        return result

    def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Generate compliance report for date range.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report
        """
        if not self.audit_logging:
            return {"status": "disabled", "reason": "AuditLogging module disabled"}

        result = self.audit_logging.get_audit_report(start_date, end_date)
        self._log_operation("get_compliance_report", result)
        return result

    # ==================== Anonymization Operations ====================

    def anonymize_dataset(
        self,
        dataset: List[Dict],
        k_value: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> Dict:
        """
        Anonymize dataset using k-anonymity and differential privacy.

        Args:
            dataset: Dataset to anonymize
            k_value: Target k-anonymity level
            epsilon: Privacy budget

        Returns:
            Anonymization result
        """
        if not self.anonymization:
            return {"status": "disabled", "reason": "Anonymization module disabled"}

        k_value = k_value or self.config.default_k_anonymity
        epsilon = epsilon or self.config.default_epsilon

        result = self.anonymization.anonymize_dataset(
            dataset, k_value, apply_dp=self.config.enable_differential_privacy, epsilon=epsilon
        )

        if self.audit_logging:
            self.audit_logging.log_event(
                "data_export",
                "system",
                f"{result['anonymized_records']} records",
                f"Dataset anonymized: k={k_value}, ε={epsilon}",
            )

        self._log_operation("anonymize_dataset", result)
        return result

    def assess_privacy_risk(self, dataset: List[Dict]) -> Dict:
        """
        Assess re-identification risk of dataset.

        Args:
            dataset: Dataset to assess

        Returns:
            Privacy risk assessment
        """
        if not self.anonymization:
            return {"status": "disabled", "reason": "Anonymization module disabled"}

        result = self.anonymization.assess_privacy(dataset)
        self._log_operation("assess_privacy_risk", result)
        return result

    # ==================== Overall Compliance Operations ====================

    def check_overall_compliance(self) -> Dict:
        """
        Check compliance status across all modules.

        Returns:
            Overall compliance status
        """
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "modules": {},
        }

        if self.gdpr:
            status["modules"]["gdpr"] = self.gdpr.check_compliance()

        if self.ccpa:
            status["modules"]["ccpa"] = self.ccpa.check_compliance()

        if self.hipaa:
            status["modules"]["hipaa"] = self.hipaa.check_compliance()

        if self.data_retention:
            status["modules"]["data_retention"] = self.data_retention.check_compliance()

        if self.audit_logging:
            status["modules"]["audit_logging"] = self.audit_logging.check_compliance()

        if self.anonymization:
            status["modules"]["anonymization"] = self.anonymization.generate_report()

        overall_status = all(
            m.get("status") == "compliant" or m.get("status") == "operational"
            for m in status["modules"].values()
        )

        status["overall_status"] = "compliant" if overall_status else "violations_detected"

        self._log_operation("check_overall_compliance", status)
        return status

    def generate_compliance_report(self) -> Dict:
        """
        Generate comprehensive compliance report.

        Returns:
            Detailed compliance report
        """
        report = {
            "report_type": "Phase 13 Comprehensive Compliance Report",
            "generated": datetime.utcnow().isoformat(),
            "configuration": {
                "gdpr_enabled": self.config.enable_gdpr,
                "ccpa_enabled": self.config.enable_ccpa,
                "hipaa_enabled": self.config.enable_hipaa,
                "data_retention_enabled": self.config.enable_data_retention,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "anonymization_enabled": self.config.enable_anonymization,
            },
            "module_status": self.check_overall_compliance(),
            "compliance_history_count": len(self.history),
        }

        return report

    def get_phase13_status(self) -> Dict:
        """
        Get comprehensive Phase 13 status.

        Returns:
            Status report
        """
        return {
            "phase": "Phase 13",
            "module": "Compliance & Privacy",
            "timestamp": datetime.utcnow().isoformat(),
            "enabled_modules": sum(
                [
                    self.config.enable_gdpr,
                    self.config.enable_ccpa,
                    self.config.enable_hipaa,
                    self.config.enable_data_retention,
                    self.config.enable_audit_logging,
                    self.config.enable_anonymization,
                ]
            ),
            "total_modules": 6,
            "overall_status": self.check_overall_compliance()["overall_status"],
            "operations_logged": len(self.history),
        }

    def _log_operation(self, operation: str, result: Dict) -> None:
        """
        Log integration operation.

        Args:
            operation: Operation name
            result: Operation result
        """
        self.history.append(
            {
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            }
        )
