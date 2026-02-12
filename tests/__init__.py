# Test suite for AegisPCAP
"""
Comprehensive test suite for AegisPCAP.

Test organization:
- tests/unit/          - Fast unit tests (no external services)
- tests/integration/   - Integration tests (API, database)
- tests/fixtures/      - Shared test fixtures and mocks

Run all tests:
    pytest

Run specific tests:
    pytest tests/unit/test_threat_intel.py
    pytest -m unit  # Only unit tests
    pytest -m integration  # Only integration tests

With coverage:
    pytest --cov=src --cov-report=html
"""
