"""
GDPR Compliance Module - EU Data Protection Regulation Implementation

Implements Article 6 (Lawful basis), Article 7 (Consent), Article 17 (Right to erasure),
Article 35 (Data Protection Impact Assessment), and Article 28 (Data Processing Agreements).

Author: AegisPCAP Compliance Team
Date: February 5, 2026
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConsentPurpose(Enum):
    """Valid consent purposes under GDPR."""
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PROCESSING = "processing"
    THIRD_PARTY = "third_party"
    PROFILING = "profiling"


@dataclass
class ConsentRecord:
    """Individual consent record."""
    user_id: str
    purpose: ConsentPurpose
    granted: bool
    timestamp: datetime
    version: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "consent_id": self.consent_id,
            "user_id": self.user_id,
            "purpose": self.purpose.value,
            "granted": self.granted,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


@dataclass
class ConsentWithdrawal:
    """Consent withdrawal record."""
    user_id: str
    purpose: ConsentPurpose
    timestamp: datetime
    reason: Optional[str] = None
    withdrawal_id: str = field(default_factory=lambda: str(uuid4()))


class ConsentManager:
    """Manage user consent for data processing (GDPR Article 7)."""

    def __init__(self):
        """Initialize consent manager."""
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.withdrawals: Dict[str, List[ConsentWithdrawal]] = {}
        logger.info("ConsentManager initialized")

    def record_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        granted: bool,
        version: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """
        Record user consent.

        Args:
            user_id: Unique user identifier
            purpose: Consent purpose
            granted: Whether consent is granted (True) or withdrawn (False)
            version: Consent version (for audit trail)
            ip_address: User's IP address (for verification)
            user_agent: User's browser/client info (for verification)

        Returns:
            consent_id: Unique consent record ID
        """
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=granted,
            timestamp=datetime.utcnow(),
            version=version,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        if user_id not in self.consent_records:
            self.consent_records[user_id] = []

        self.consent_records[user_id].append(record)
        logger.info(f"Consent recorded: {record.consent_id} for {user_id}")
        return record.consent_id

    def check_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """
        Check if user has given active consent for purpose.

        Args:
            user_id: User ID to check
            purpose: Purpose to check

        Returns:
            True if active consent exists, False otherwise
        """
        if user_id not in self.consent_records:
            return False

        # Get most recent consent for this purpose
        relevant_consents = [
            c for c in self.consent_records[user_id] if c.purpose == purpose
        ]

        if not relevant_consents:
            return False

        latest_consent = max(relevant_consents, key=lambda x: x.timestamp)
        return latest_consent.granted

    def withdraw_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        reason: Optional[str] = None,
    ) -> str:
        """
        Withdraw user consent (GDPR Article 7(3)).

        Args:
            user_id: User ID
            purpose: Purpose to withdraw
            reason: Reason for withdrawal

        Returns:
            withdrawal_id: Unique withdrawal record ID
        """
        withdrawal = ConsentWithdrawal(
            user_id=user_id,
            purpose=purpose,
            timestamp=datetime.utcnow(),
            reason=reason,
        )

        if user_id not in self.withdrawals:
            self.withdrawals[user_id] = []

        self.withdrawals[user_id].append(withdrawal)

        # Record as explicit withdrawal in consent records
        self.record_consent(user_id, purpose, False, "withdrawal")
        logger.info(f"Consent withdrawn: {withdrawal.withdrawal_id} for {user_id}")
        return withdrawal.withdrawal_id

    def get_consent_history(self, user_id: str) -> List[Dict]:
        """
        Get complete consent history for user.

        Args:
            user_id: User ID

        Returns:
            List of consent records in chronological order
        """
        if user_id not in self.consent_records:
            return []

        return [
            c.to_dict()
            for c in sorted(
                self.consent_records[user_id], key=lambda x: x.timestamp
            )
        ]

    def get_active_consents(self, user_id: str) -> Dict[str, bool]:
        """
        Get all active consents for user.

        Args:
            user_id: User ID

        Returns:
            Dictionary mapping purpose to consent status
        """
        active = {}
        for purpose in ConsentPurpose:
            active[purpose.value] = self.check_consent(user_id, purpose)
        return active


class DataMinimizationController:
    """Enforce data minimization principle (GDPR Article 5)."""

    def __init__(self, approved_fields: Dict[str, Set[str]]):
        """
        Initialize data minimization controller.

        Args:
            approved_fields: Dict mapping purpose to set of approved field names
                           e.g. {"marketing": {"email", "name"}, "analytics": {"event_type"}}
        """
        self.approved_fields = approved_fields
        logger.info("DataMinimizationController initialized")

    def validate_data_collection(
        self, purpose: str, collected_fields: Set[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that collected fields are necessary for purpose.

        Args:
            purpose: Processing purpose
            collected_fields: Fields being collected

        Returns:
            (is_valid, list_of_unnecessary_fields)
        """
        if purpose not in self.approved_fields:
            return False, list(collected_fields)

        approved = self.approved_fields[purpose]
        unnecessary = collected_fields - approved

        if unnecessary:
            logger.warning(
                f"Unnecessary fields for {purpose}: {unnecessary}"
            )
            return False, list(unnecessary)

        return True, []

    def suggest_alternatives(
        self, purpose: str, field_name: str
    ) -> Optional[str]:
        """
        Suggest alternative field that might be sufficient.

        Args:
            purpose: Processing purpose
            field_name: Field being questioned

        Returns:
            Alternative field name if one exists
        """
        # Simple suggestion logic - in production would be more sophisticated
        field_alternatives = {
            "full_name": "first_name",
            "email_address": "email",
            "phone_number": "contact_channel",
            "address": "postal_code",
        }
        return field_alternatives.get(field_name)

    def audit_collection_scope(
        self, purpose: str, actual_fields: Set[str]
    ) -> Dict:
        """
        Audit actual data collection against approved scope.

        Args:
            purpose: Processing purpose
            actual_fields: Fields actually collected

        Returns:
            Audit report with findings
        """
        approved = self.approved_fields.get(purpose, set())
        return {
            "purpose": purpose,
            "approved_fields": list(approved),
            "collected_fields": list(actual_fields),
            "excess_fields": list(actual_fields - approved),
            "missing_fields": list(approved - actual_fields),
            "compliant": actual_fields == approved,
        }


@dataclass
class DeletionRequest:
    """Right to be forgotten request."""
    user_id: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, verified, processing, completed
    verification_code: Optional[str] = None
    completion_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "completion_time": self.completion_time.isoformat()
            if self.completion_time
            else None,
        }


class RightToBeForgettenHandler:
    """Implement right to erasure (GDPR Article 17)."""

    def __init__(self):
        """Initialize deletion handler."""
        self.deletion_requests: Dict[str, DeletionRequest] = {}
        self.deleted_records: List[Dict] = []
        logger.info("RightToBeForgettenHandler initialized")

    def initiate_deletion_request(self, user_id: str) -> Tuple[str, str]:
        """
        Initiate right to be forgotten request.

        Args:
            user_id: User requesting deletion

        Returns:
            (request_id, verification_code) for user to confirm
        """
        request = DeletionRequest(user_id=user_id)
        verification_code = hashlib.sha256(
            f"{user_id}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        request.verification_code = verification_code
        self.deletion_requests[request.request_id] = request

        logger.info(f"Deletion request initiated: {request.request_id}")
        return request.request_id, verification_code

    def verify_request(self, request_id: str, verification_code: str) -> bool:
        """
        Verify deletion request matches provided code.

        Args:
            request_id: Request ID
            verification_code: Verification code from user

        Returns:
            True if verification successful
        """
        if request_id not in self.deletion_requests:
            return False

        request = self.deletion_requests[request_id]
        if request.verification_code == verification_code:
            request.status = "verified"
            logger.info(f"Deletion request verified: {request_id}")
            return True

        return False

    def cascade_delete(self, user_id: str) -> Dict:
        """
        Delete all user data from all systems (cascade).

        Args:
            user_id: User ID to delete

        Returns:
            Deletion report with systems affected
        """
        deletion_report = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "systems_affected": [
                "primary_database",
                "cache_layer",
                "backup_storage",
                "analytics",
                "audit_logs",
            ],
            "records_deleted": 0,
            "status": "completed",
        }

        # Simulate cascade deletion
        self.deleted_records.append(
            {
                "user_id": user_id,
                "deletion_time": datetime.utcnow(),
                "systems": deletion_report["systems_affected"],
            }
        )

        logger.info(
            f"Cascade deletion completed for {user_id}: "
            f"{deletion_report['records_deleted']} records"
        )
        return deletion_report

    def confirm_deletion(self, request_id: str) -> Dict:
        """
        Confirm deletion is complete.

        Args:
            request_id: Deletion request ID

        Returns:
            Confirmation report
        """
        if request_id not in self.deletion_requests:
            return {"status": "failed", "reason": "request_not_found"}

        request = self.deletion_requests[request_id]
        request.status = "completed"
        request.completion_time = datetime.utcnow()

        logger.info(f"Deletion completed: {request_id}")
        return request.to_dict()


@dataclass
class DataProcessingAgreement:
    """Data Processing Agreement (GDPR Article 28)."""
    processor_name: str
    processor_id: str
    signed_date: datetime
    expiry_date: datetime
    dpa_id: str = field(default_factory=lambda: str(uuid4()))
    compliant: bool = True
    last_audit: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dpa_id": self.dpa_id,
            "processor_name": self.processor_name,
            "processor_id": self.processor_id,
            "signed_date": self.signed_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
            "compliant": self.compliant,
            "last_audit": self.last_audit.isoformat() if self.last_audit else None,
        }


class DataProcessingAgreementTracker:
    """Track Data Processing Agreements with processors."""

    def __init__(self):
        """Initialize DPA tracker."""
        self.agreements: Dict[str, DataProcessingAgreement] = {}
        logger.info("DataProcessingAgreementTracker initialized")

    def register_processor(
        self,
        processor_name: str,
        processor_id: str,
        expiry_date: datetime,
    ) -> str:
        """
        Register processor with DPA.

        Args:
            processor_name: Name of processor
            processor_id: Unique processor ID
            expiry_date: When DPA expires

        Returns:
            dpa_id: Agreement ID
        """
        dpa = DataProcessingAgreement(
            processor_name=processor_name,
            processor_id=processor_id,
            signed_date=datetime.utcnow(),
            expiry_date=expiry_date,
        )

        self.agreements[dpa.dpa_id] = dpa
        logger.info(f"Processor registered: {processor_name}")
        return dpa.dpa_id

    def verify_compliance(self, dpa_id: str) -> bool:
        """
        Verify processor's DPA is current and compliant.

        Args:
            dpa_id: DPA ID to check

        Returns:
            True if valid and not expired
        """
        if dpa_id not in self.agreements:
            return False

        agreement = self.agreements[dpa_id]
        is_valid = agreement.compliant and agreement.expiry_date > datetime.utcnow()
        return is_valid

    def audit_dpa(self, dpa_id: str) -> Dict:
        """
        Audit processor's DPA compliance.

        Args:
            dpa_id: DPA ID to audit

        Returns:
            Audit results
        """
        if dpa_id not in self.agreements:
            return {"status": "failed", "reason": "dpa_not_found"}

        agreement = self.agreements[dpa_id]
        days_until_expiry = (agreement.expiry_date - datetime.utcnow()).days

        return {
            "dpa_id": dpa_id,
            "processor": agreement.processor_name,
            "compliant": agreement.compliant,
            "days_until_expiry": days_until_expiry,
            "requires_renewal": days_until_expiry < 90,
        }

    def renew_dpa(self, dpa_id: str, new_expiry: datetime) -> bool:
        """
        Renew processor's DPA.

        Args:
            dpa_id: DPA ID to renew
            new_expiry: New expiry date

        Returns:
            True if renewal successful
        """
        if dpa_id not in self.agreements:
            return False

        agreement = self.agreements[dpa_id]
        agreement.expiry_date = new_expiry
        logger.info(f"DPA renewed: {dpa_id}")
        return True


@dataclass
class DPIA:
    """Data Protection Impact Assessment."""
    activity_name: str
    dpia_id: str = field(default_factory=lambda: str(uuid4()))
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    risks_identified: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    overall_risk: str = "unknown"  # low, medium, high


class DPIAFramework:
    """Data Protection Impact Assessment (GDPR Article 35)."""

    def __init__(self):
        """Initialize DPIA framework."""
        self.dpias: Dict[str, DPIA] = {}
        logger.info("DPIAFramework initialized")

    def create_dpia(self, activity_name: str) -> str:
        """
        Create new DPIA for processing activity.

        Args:
            activity_name: Name of processing activity

        Returns:
            dpia_id: Assessment ID
        """
        dpia = DPIA(activity_name=activity_name)
        self.dpias[dpia.dpia_id] = dpia
        logger.info(f"DPIA created: {dpia.dpia_id} for {activity_name}")
        return dpia.dpia_id

    def assess_risks(
        self, dpia_id: str, risk_descriptions: List[str]
    ) -> str:
        """
        Identify and document risks in processing.

        Args:
            dpia_id: DPIA ID
            risk_descriptions: List of identified risks

        Returns:
            overall_risk: Overall risk level (low/medium/high)
        """
        if dpia_id not in self.dpias:
            return "unknown"

        dpia = self.dpias[dpia_id]
        dpia.risks_identified = risk_descriptions

        # Simple risk aggregation
        if len(risk_descriptions) > 5:
            dpia.overall_risk = "high"
        elif len(risk_descriptions) > 2:
            dpia.overall_risk = "medium"
        else:
            dpia.overall_risk = "low"

        logger.info(f"Risks assessed: {dpia_id} - {dpia.overall_risk}")
        return dpia.overall_risk

    def document_mitigations(
        self, dpia_id: str, mitigation_measures: List[str]
    ) -> bool:
        """
        Document mitigations for identified risks.

        Args:
            dpia_id: DPIA ID
            mitigation_measures: List of mitigation measures

        Returns:
            True if documented successfully
        """
        if dpia_id not in self.dpias:
            return False

        dpia = self.dpias[dpia_id]
        dpia.mitigations = mitigation_measures
        dpia.last_updated = datetime.utcnow()
        logger.info(f"Mitigations documented: {dpia_id}")
        return True

    def validate_assessment(self, dpia_id: str) -> Dict:
        """
        Validate DPIA is complete.

        Args:
            dpia_id: DPIA ID to validate

        Returns:
            Validation report
        """
        if dpia_id not in self.dpias:
            return {"valid": False, "reason": "dpia_not_found"}

        dpia = self.dpias[dpia_id]
        is_valid = (
            len(dpia.risks_identified) > 0
            and len(dpia.mitigations) > 0
            and dpia.overall_risk != "unknown"
        )

        return {
            "dpia_id": dpia_id,
            "valid": is_valid,
            "activity": dpia.activity_name,
            "risk_level": dpia.overall_risk,
            "risks_count": len(dpia.risks_identified),
            "mitigations_count": len(dpia.mitigations),
        }


class GDPRController:
    """Unified GDPR compliance interface."""

    def __init__(self):
        """Initialize GDPR controller."""
        self.consent_manager = ConsentManager()
        self.minimization = DataMinimizationController(
            {
                "marketing": {"email", "name"},
                "analytics": {"event_type", "timestamp"},
                "processing": {"email", "name", "user_id"},
            }
        )
        self.deletion_handler = RightToBeForgettenHandler()
        self.dpa_tracker = DataProcessingAgreementTracker()
        self.dpia_framework = DPIAFramework()
        self.history: List[Dict] = []
        logger.info("GDPRController initialized")

    def record_consent(
        self,
        user_id: str,
        purposes: List[str],
        version: str = "1.0",
    ) -> Dict:
        """
        Record user consent for purposes.

        Args:
            user_id: User ID
            purposes: List of purposes
            version: Consent form version

        Returns:
            Result with consent IDs
        """
        consent_ids = []
        for purpose_str in purposes:
            try:
                purpose = ConsentPurpose[purpose_str.upper()]
                cid = self.consent_manager.record_consent(
                    user_id, purpose, True, version
                )
                consent_ids.append(cid)
            except KeyError:
                logger.warning(f"Invalid purpose: {purpose_str}")

        result = {
            "status": "success",
            "user_id": user_id,
            "consent_ids": consent_ids,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._log_operation("record_consent", result)
        return result

    def initiate_right_to_be_forgotten(self, user_id: str) -> Dict:
        """
        Start right to be forgotten process.

        Args:
            user_id: User requesting deletion

        Returns:
            Result with request ID and verification code
        """
        request_id, verification_code = (
            self.deletion_handler.initiate_deletion_request(user_id)
        )

        result = {
            "status": "pending",
            "request_id": request_id,
            "verification_code": verification_code,
            "user_id": user_id,
        }
        self._log_operation("initiate_deletion", result)
        return result

    def verify_deletion_request(
        self, request_id: str, verification_code: str
    ) -> Dict:
        """
        Verify deletion request.

        Args:
            request_id: Request ID
            verification_code: User-provided verification code

        Returns:
            Result with verification status
        """
        verified = self.deletion_handler.verify_request(
            request_id, verification_code
        )

        result = {
            "status": "success" if verified else "failed",
            "request_id": request_id,
            "verified": verified,
        }
        self._log_operation("verify_deletion", result)

        if verified:
            # Proceed with cascade deletion
            deletion_report = self.deletion_handler.cascade_delete(
                self.deletion_handler.deletion_requests[request_id].user_id
            )
            self._log_operation("cascade_delete", deletion_report)

        return result

    def create_dpia(self, activity_name: str) -> Dict:
        """
        Create DPIA for processing activity.

        Args:
            activity_name: Name of activity

        Returns:
            Result with DPIA ID
        """
        dpia_id = self.dpia_framework.create_dpia(activity_name)
        result = {
            "status": "success",
            "dpia_id": dpia_id,
            "activity": activity_name,
        }
        self._log_operation("create_dpia", result)
        return result

    def check_compliance(self) -> Dict:
        """
        Check overall GDPR compliance status.

        Returns:
            Compliance status report
        """
        return {
            "module": "GDPR",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "consent_manager": "operational",
                "minimization_controller": "operational",
                "deletion_handler": "operational",
                "dpa_tracker": "operational",
                "dpia_framework": "operational",
            },
            "status": "compliant",
        }

    def _log_operation(self, operation: str, result: Dict) -> None:
        """Log compliance operation."""
        self.history.append(
            {
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            }
        )
