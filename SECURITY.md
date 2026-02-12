# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please report security issues responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email: **security@aegispcap.org**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Best effort

### Disclosure Policy

- We will acknowledge your report within 48 hours
- We will provide regular updates on our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We request that you do not publicly disclose the vulnerability until we have released a fix

## Security Best Practices

### For Users

- Keep AegisPCAP updated to the latest version
- Use strong authentication credentials
- Enable TLS/SSL for all connections
- Regularly review audit logs
- Follow principle of least privilege

### For Contributors

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Follow secure coding practices
- Run security scans before submitting PRs
- Review dependencies for known vulnerabilities

## Security Features

AegisPCAP includes:
- JWT-based authentication
- Role-based access control (RBAC)
- TLS 1.3 encryption
- Automatic PII anonymization
- Comprehensive audit logging
- Rate limiting and DDoS protection

## Security Audits

We conduct regular security audits and welcome external security reviews.

## Bug Bounty

We currently do not have a formal bug bounty program, but we greatly appreciate responsible disclosure and will acknowledge contributors in our security advisories.

## Contact

For security-related questions: security@aegispcap.org
For general questions: See [CONTRIBUTING.md](CONTRIBUTING.md)
