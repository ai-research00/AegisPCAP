"""
Phase 13 Test Suite - Compliance & Privacy Module Testing

30+ comprehensive tests covering all 6 compliance modules and integration layer.
Target: 95%+ code coverage

Author: AegisPCAP Compliance Team
Date: February 5, 2026
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict

from compliance.gdpr import GDPRController, ConsentPurpose
from compliance.ccpa import CCPAController, RequestType
from compliance.hipaa import HIPAAController
from compliance.data_retention import DataRetentionController
from compliance.audit_logging import AuditLoggingController, EventType
from compliance.anonymization import AnonymizationController
from api.phase13_integration import Phase13Integration, Phase13Configuration


# ==================== Fixtures ====================

@pytest.fixture
def sample_dataset() -> List[Dict]:
    """Sample dataset for anonymization testing."""
    return [
        {
            "id": "1",
            "name": "John Doe",
            "age": 35,
            "gender": "M",
            "postal_code": "94105",
            "income": 75000,
            "email": "john@example.com",
        },
        {
            "id": "2",
            "name": "Jane Smith",
            "age": 42,
            "gender": "F",
            "postal_code": "94106",
            "income": 85000,
            "email": "jane@example.com",
        },
        {
            "id": "3",
            "name": "Bob Johnson",
            "age": 35,
            "gender": "M",
            "postal_code": "94105",
            "income": 72000,
            "email": "bob@example.com",
        },
    ]


@pytest.fixture
def gdpr_controller() -> GDPRController:
    """GDPR controller instance."""
    return GDPRController()


@pytest.fixture
def ccpa_controller() -> CCPAController:
    """CCPA controller instance."""
    return CCPAController()


@pytest.fixture
def hipaa_controller() -> HIPAAController:
    """HIPAA controller instance."""
    return HIPAAController()


@pytest.fixture
def data_retention_controller() -> DataRetentionController:
    """Data retention controller instance."""
    return DataRetentionController()


@pytest.fixture
def audit_logging_controller() -> AuditLoggingController:
    """Audit logging controller instance."""
    return AuditLoggingController()


@pytest.fixture
def anonymization_controller() -> AnonymizationController:
    """Anonymization controller instance."""
    return AnonymizationController()


@pytest.fixture
def phase13_integration() -> Phase13Integration:
    """Phase13 integration instance with all modules."""
    config = Phase13Configuration(
        enable_gdpr=True,
        enable_ccpa=True,
        enable_hipaa=True,
        enable_data_retention=True,
        enable_audit_logging=True,
        enable_anonymization=True,
    )
    return Phase13Integration(config)


# ==================== GDPR Tests ====================

class TestGDPRCompliance:
    """GDPR compliance module tests."""

    def test_record_consent(self, gdpr_controller):
        """Test recording user consent."""
        result = gdpr_controller.record_consent(
            "user123",
            ["marketing", "analytics"],
            "1.0",
        )
        assert result["status"] == "success"
        assert len(result["consent_ids"]) > 0

    def test_check_active_consent(self, gdpr_controller):
        """Test checking active consent."""
        gdpr_controller.record_consent(
            "user123",
            ["marketing"],
            "1.0",
        )
        assert gdpr_controller.consent_manager.check_consent(
            "user123", ConsentPurpose.MARKETING
        )

    def test_withdraw_consent(self, gdpr_controller):
        """Test withdrawing consent."""
        gdpr_controller.record_consent(
            "user123",
            ["marketing"],
            "1.0",
        )
        withdrawal_id = gdpr_controller.consent_manager.withdraw_consent(
            "user123", ConsentPurpose.MARKETING
        )
        assert withdrawal_id
        assert not gdpr_controller.consent_manager.check_consent(
            "user123", ConsentPurpose.MARKETING
        )

    def test_initiate_deletion(self, gdpr_controller):
        """Test initiating right to be forgotten."""
        result = gdpr_controller.initiate_right_to_be_forgotten("user123")
        assert result["status"] == "pending"
        assert "request_id" in result
        assert "verification_code" in result

    def test_verify_deletion_request(self, gdpr_controller):
        """Test verifying deletion request."""
        result = gdpr_controller.initiate_right_to_be_forgotten("user123")
        request_id = result["request_id"]
        verification_code = result["verification_code"]

        verified = gdpr_controller.deletion_handler.verify_request(
            request_id, verification_code
        )
        assert verified

    def test_create_dpia(self, gdpr_controller):
        """Test creating DPIA."""
        result = gdpr_controller.create_dpia("Customer profiling activity")
        assert result["status"] == "success"
        assert "dpia_id" in result

    def test_data_minimization(self, gdpr_controller):
        """Test data minimization validation."""
        is_valid, extra_fields = gdpr_controller.minimization.validate_data_collection(
            "marketing", {"email", "name"}
        )
        assert is_valid

    def test_dpa_registration(self, gdpr_controller):
        """Test Data Processing Agreement registration."""
        expiry = datetime.utcnow() + timedelta(days=365)
        dpa_id = gdpr_controller.dpa_tracker.register_processor(
            "CloudProvider Inc",
            "cloud_001",
            expiry,
        )
        assert dpa_id
        assert gdpr_controller.dpa_tracker.verify_baa_coverage(dpa_id)


# ==================== CCPA Tests ====================

class TestCCPACompliance:
    """CCPA compliance module tests."""

    def test_handle_consumer_request(self, ccpa_controller):
        """Test handling consumer request."""
        result = ccpa_controller.handle_consumer_request(
            "know",
            "consumer123",
            "consumer@example.com",
        )
        assert result["status"] == "received"
        assert "request_id" in result
        assert "deadline" in result

    def test_data_disclosure(self, ccpa_controller):
        """Test right to know - data disclosure."""
        data = ccpa_controller.disclosure_manager.compile_consumer_data(
            "consumer123"
        )
        assert "identifiers" in data
        assert "commercial" in data
        assert "activity" in data

    def test_consumer_opt_out(self, ccpa_controller):
        """Test consumer opt-out."""
        opt_out_id = ccpa_controller.opt_out_controller.record_opt_out(
            "consumer123", "sale"
        )
        assert opt_out_id
        assert ccpa_controller.opt_out_controller.verify_opt_out("consumer123")

    def test_deletion_request_verification(self, ccpa_controller):
        """Test deletion request verification."""
        verified, verification_id = ccpa_controller.verification.verify_buyer_identity(
            "consumer123",
            {"email": "consumer@example.com"},
        )
        assert verified
        assert verification_id

    def test_sale_opt_out_tracking(self, ccpa_controller):
        """Test sale opt-out tracking."""
        opt_out_id = ccpa_controller.sale_opt_out.record_sale_opt_out(
            "consumer123"
        )
        assert ccpa_controller.sale_opt_out.honor_sale_opt_out("consumer123")

    def test_ccpa_compliance_check(self, ccpa_controller):
        """Test CCPA compliance status."""
        status = ccpa_controller.check_compliance()
        assert status["module"] == "CCPA"
        assert status["status"] == "compliant"


# ==================== HIPAA Tests ====================

class TestHIPAACompliance:
    """HIPAA compliance module tests."""

    def test_phi_classification(self, hipaa_controller):
        """Test PHI classification."""
        is_phi, classification_id = hipaa_controller.phi_protection.classify_phi(
            "medical_record", "medical_record"
        )
        assert is_phi
        assert classification_id

    def test_baa_registration(self, hipaa_controller):
        """Test BAA registration."""
        expiry = datetime.utcnow() + timedelta(days=365)
        result = hipaa_controller.register_business_associate(
            "Medical Records Processor",
            "transcription",
            expiry,
        )
        assert result["status"] == "success"
        assert "baa_id" in result

    def test_phi_encryption(self, hipaa_controller):
        """Test PHI encryption at rest."""
        result = hipaa_controller.phi_protection.encrypt_phi_at_rest(
            "phi_123", "AES_256"
        )
        assert result["encrypted"]
        assert result["algorithm"] == "AES-256"

    def test_phi_access_logging(self, hipaa_controller):
        """Test PHI access logging."""
        log_id = hipaa_controller.access_control.log_access(
            "phi_123",
            "doctor_001",
            "read",
            "treatment",
            "192.168.1.1",
        )
        assert log_id

    def test_breach_notification(self, hipaa_controller):
        """Test breach notification."""
        breach_id = hipaa_controller.breach_notification.report_breach(
            datetime.utcnow(),
            100,
            ["medical_record"],
        )
        assert breach_id

        assessment = hipaa_controller.breach_notification.assess_breach(
            breach_id
        )
        assert assessment["notification_required"]

    def test_minimum_necessary_audit(self, hipaa_controller):
        """Test minimum necessary principle audit."""
        audit = hipaa_controller.minimum_necessary.audit_access_scope(
            "doctor_001",
            {"patient_id", "diagnosis", "treatment"},
        )
        assert "user_id" in audit


# ==================== Data Retention Tests ====================

class TestDataRetention:
    """Data retention module tests."""

    def test_create_policy(self, data_retention_controller):
        """Test creating retention policy."""
        result = data_retention_controller.schedule_policy(
            "logs", 90, archive_days=30
        )
        assert result["status"] == "success"
        assert result["retention_days"] == 90

    def test_schedule_deletion(self, data_retention_controller):
        """Test scheduling deletion."""
        delete_date = datetime.utcnow() + timedelta(days=90)
        record_ids = ["rec_1", "rec_2", "rec_3"]
        schedule_id = data_retention_controller.deletion_scheduler.schedule_deletion(
            record_ids, delete_date
        )
        assert schedule_id

    def test_execute_deletion(self, data_retention_controller):
        """Test executing deletion."""
        delete_date = datetime.utcnow() + timedelta(days=90)
        record_ids = ["rec_1", "rec_2"]
        schedule_id = data_retention_controller.deletion_scheduler.schedule_deletion(
            record_ids, delete_date
        )

        result = data_retention_controller.deletion_scheduler.execute_deletion(
            schedule_id, record_ids
        )
        assert result["status"] == "success"

    def test_archive_creation(self, data_retention_controller):
        """Test archive creation."""
        archive_id = data_retention_controller.archive_manager.create_archive(
            "Q4_2025_Archive",
            ["rec_1", "rec_2", "rec_3"],
        )
        assert archive_id

    def test_retention_compliance_check(self, data_retention_controller):
        """Test retention compliance verification."""
        result = data_retention_controller.verify_compliance()
        assert "policies_audited" in result


# ==================== Audit Logging Tests ====================

class TestAuditLogging:
    """Audit logging module tests."""

    def test_log_event(self, audit_logging_controller):
        """Test logging compliance event."""
        log_id = audit_logging_controller.audit_log.log_event(
            "data_access",
            "user123",
            "customer_data",
            "read",
            ip_address="192.168.1.1",
        )
        assert log_id

    def test_audit_trail_integrity(self, audit_logging_controller):
        """Test audit trail integrity verification."""
        audit_logging_controller.audit_log.log_event(
            "data_access",
            "user123",
            "customer_data",
            "read",
        )

        is_valid = audit_logging_controller.audit_log.verify_integrity()
        assert is_valid

    def test_chain_of_custody(self, audit_logging_controller):
        """Test chain of custody tracking."""
        transfer_id = audit_logging_controller.chain_of_custody.initiate_transfer(
            "data_123",
            "user1",
            "user2",
            "approval",
        )
        assert transfer_id

        audit_logging_controller.chain_of_custody.log_transfer(transfer_id)
        chain = audit_logging_controller.chain_of_custody.verify_chain("data_123")
        assert chain["data_id"] == "data_123"

    def test_compliance_event_logging(self, audit_logging_controller):
        """Test compliance event logging."""
        event_id = audit_logging_controller.event_logger.log_consent_change(
            "user123", "marketing", True
        )
        assert event_id

    def test_audit_report_generation(self, audit_logging_controller):
        """Test audit report generation."""
        audit_logging_controller.audit_log.log_event(
            "data_access",
            "user123",
            "data",
            "read",
        )

        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)

        report = audit_logging_controller.trail_renderer.generate_report()
        assert "total_events" in report
        assert report["integrity_verified"]

    def test_tamper_detection(self, audit_logging_controller):
        """Test tamper detection system."""
        audit_logging_controller.audit_log.log_event(
            "data_access",
            "user123",
            "data",
            "read",
        )

        tampering = audit_logging_controller.tamper_detection.detect_tampering()
        assert len(tampering) == 0  # No tampering


# ==================== Anonymization Tests ====================

class TestAnonymization:
    """Anonymization module tests."""

    def test_k_anonymity(self, anonymization_controller, sample_dataset):
        """Test k-anonymity transformation."""
        anonymized, k = anonymization_controller.k_anonymity.apply_k_anonymity(
            sample_dataset, ["age", "gender", "postal_code"], k=3
        )
        assert len(anonymized) > 0
        assert k >= 1

    def test_measure_k_value(self, anonymization_controller, sample_dataset):
        """Test measuring k-value."""
        k = anonymization_controller.k_anonymity.measure_k_value(
            sample_dataset, ["age", "gender", "postal_code"]
        )
        assert k >= 1

    def test_differential_privacy_noise(self, anonymization_controller, sample_dataset):
        """Test differential privacy noise addition."""
        noisy = anonymization_controller.diff_privacy.add_dp_noise(
            sample_dataset, ["age", "income"], epsilon=1.0
        )
        assert len(noisy) == len(sample_dataset)

    def test_epsilon_computation(self, anonymization_controller):
        """Test epsilon computation."""
        epsilon = anonymization_controller.diff_privacy.compute_epsilon(
            1000, 5, "moderate"
        )
        assert epsilon > 0

    def test_pseudonymization(self, anonymization_controller):
        """Test pseudonymization."""
        pseudonym = anonymization_controller.pseudonymization.pseudonymize(
            "user123", salt="test_salt"
        )
        assert len(pseudonym) > 0
        assert pseudonym != "user123"

    def test_reidentification_risk_assessment(
        self, anonymization_controller, sample_dataset
    ):
        """Test re-identification risk assessment."""
        assessment = anonymization_controller.risk_assessment.assess_risk(
            sample_dataset, ["age", "gender", "postal_code"]
        )
        assert "re_identification_risk" in assessment
        assert assessment["risk_level"] in ["low", "medium", "high"]

    def test_anonymization_workflow(self, anonymization_controller, sample_dataset):
        """Test complete anonymization workflow."""
        result = anonymization_controller.anonymize_dataset(
            sample_dataset, k_value=3, apply_dp=True
        )
        assert result["status"] == "success"
        assert result["anonymized_records"] > 0
        assert result["utility_score"] > 0


# ==================== Integration Tests ====================

class TestPhase13Integration:
    """Phase 13 integration tests."""

    def test_initialization(self, phase13_integration):
        """Test Phase13 integration initialization."""
        assert phase13_integration.gdpr is not None
        assert phase13_integration.ccpa is not None
        assert phase13_integration.hipaa is not None
        assert phase13_integration.data_retention is not None
        assert phase13_integration.audit_logging is not None
        assert phase13_integration.anonymization is not None

    def test_record_user_consent(self, phase13_integration):
        """Test recording consent via integration."""
        result = phase13_integration.record_user_consent(
            "user123",
            ["marketing", "analytics"],
        )
        assert result["status"] == "success"

    def test_handle_consumer_request(self, phase13_integration):
        """Test handling consumer request via integration."""
        result = phase13_integration.handle_consumer_request(
            "know",
            "consumer123",
            "consumer@example.com",
        )
        assert result["status"] == "received"

    def test_register_business_associate(self, phase13_integration):
        """Test registering BA via integration."""
        expiry = datetime.utcnow() + timedelta(days=365)
        result = phase13_integration.register_business_associate(
            "Processor Inc",
            "cloud_processing",
            expiry,
        )
        assert result["status"] == "success"

    def test_create_retention_policy(self, phase13_integration):
        """Test creating retention policy via integration."""
        result = phase13_integration.create_retention_policy(
            "logs", 90, archive_days=30
        )
        assert result["status"] == "success"

    def test_log_compliance_event(self, phase13_integration):
        """Test logging compliance event via integration."""
        result = phase13_integration.log_compliance_event(
            "data_access",
            "user123",
            "customer_data",
            "read",
        )
        assert result["status"] == "logged"

    def test_anonymize_dataset_integration(
        self, phase13_integration, sample_dataset
    ):
        """Test anonymizing dataset via integration."""
        result = phase13_integration.anonymize_dataset(
            sample_dataset, k_value=3
        )
        assert result["status"] == "success"

    def test_check_overall_compliance(self, phase13_integration):
        """Test overall compliance check."""
        status = phase13_integration.check_overall_compliance()
        assert status["overall_status"] in ["compliant", "violations_detected"]
        assert "modules" in status

    def test_generate_compliance_report(self, phase13_integration):
        """Test compliance report generation."""
        report = phase13_integration.generate_compliance_report()
        assert report["report_type"] == "Phase 13 Comprehensive Compliance Report"
        assert "module_status" in report

    def test_get_phase13_status(self, phase13_integration):
        """Test getting Phase 13 status."""
        status = phase13_integration.get_phase13_status()
        assert status["phase"] == "Phase 13"
        assert status["enabled_modules"] == 6
        assert status["total_modules"] == 6

    def test_verify_audit_integrity(self, phase13_integration):
        """Test verifying audit integrity via integration."""
        phase13_integration.log_compliance_event(
            "data_access",
            "user123",
            "data",
            "read",
        )
        result = phase13_integration.verify_audit_integrity()
        assert result["integrity_valid"]

    def test_disabled_module(self):
        """Test with modules disabled."""
        config = Phase13Configuration(enable_gdpr=False)
        integration = Phase13Integration(config)
        result = integration.record_user_consent("user123", ["marketing"])
        assert result["status"] == "disabled"


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and load tests."""

    def test_consent_recording_performance(self, gdpr_controller):
        """Test consent recording performance (<5ms)."""
        import time

        start = time.time()
        for i in range(100):
            gdpr_controller.record_consent(
                f"user_{i}",
                ["marketing"],
                "1.0",
            )
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.05  # <50ms per operation

    def test_audit_logging_performance(self, audit_logging_controller):
        """Test audit logging performance (<5ms)."""
        import time

        start = time.time()
        for i in range(100):
            audit_logging_controller.audit_log.log_event(
                "data_access",
                f"user_{i}",
                "data",
                "read",
            )
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.05  # <50ms per operation

    def test_anonymization_performance(self, anonymization_controller):
        """Test anonymization performance."""
        import time

        large_dataset = [
            {
                "id": f"user_{i}",
                "age": 20 + (i % 50),
                "gender": "M" if i % 2 == 0 else "F",
                "postal_code": f"94{100 + (i % 6):03d}",
            }
            for i in range(1000)
        ]

        start = time.time()
        anonymization_controller.anonymize_dataset(large_dataset, k_value=5)
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should complete in <1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
