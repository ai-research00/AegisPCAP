# Contributing to AegisPCAP

Thank you for your interest in contributing to AegisPCAP! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AegisPCAP.git
   cd AegisPCAP
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ai-research00/AegisPCAP.git
   ```

## Development Setup

### Prerequisites
- Python 3.9 or higher
- PostgreSQL 14+ (for database features)
- Redis 7+ (for caching)
- Node.js 18+ (for frontend development)

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests** to verify setup:
   ```bash
   pytest tests/
   ```

## How to Contribute

### Reporting Bugs
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include detailed steps to reproduce
- Provide environment information
- Add relevant logs or screenshots

### Suggesting Features
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the problem it solves
- Describe your proposed solution
- Consider implementation complexity

### Asking Questions
- Use the [Question template](.github/ISSUE_TEMPLATE/question.md)
- Check existing issues and documentation first
- Provide context about what you're trying to accomplish

### Contributing Code

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style guidelines

3. **Write tests** for your changes

4. **Run the test suite**:
   ```bash
   pytest tests/ --cov=src
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Code Style Guidelines

### Python Code
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black src/ tests/
  ```
- Use [isort](https://pycqa.github.io/isort/) for import sorting:
  ```bash
  isort src/ tests/
  ```
- Use type hints for function signatures
- Write docstrings for all public functions/classes (Google style)

### TypeScript/React Code
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use TypeScript strict mode
- Use functional components with hooks
- Write JSDoc comments for complex functions

### General Guidelines
- Keep functions small and focused (< 50 lines)
- Use descriptive variable and function names
- Add comments for complex logic
- Avoid premature optimization
- Write self-documenting code

## Testing Requirements

### Unit Tests
- Write unit tests for all new functions/classes
- Aim for >80% code coverage
- Use pytest for Python tests
- Use Jest for TypeScript/React tests

### Integration Tests
- Add integration tests for new features
- Test API endpoints with FastAPI TestClient
- Test database interactions

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run frontend tests
cd frontend && npm test
```

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for your changes
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Fill out the PR template** completely
6. **Request review** from maintainers
7. **Address review feedback** promptly
8. **Squash commits** if requested

### PR Review Criteria
- Code follows style guidelines
- Tests are included and passing
- Documentation is updated
- No breaking changes (or clearly documented)
- Commit messages are clear
- PR description is complete

## Community

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions

### Recognition
Contributors are recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- GitHub contributors page

### Maintainer Response Time
- Issues: Within 48 hours
- Pull Requests: Within 1 week
- Security Issues: Within 24 hours

## Development Workflow

### Branch Strategy
- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

### Release Process
1. Features merged to `develop`
2. Testing on `develop` branch
3. Release candidate created
4. Merge to `main` with version tag
5. Automated deployment

## Additional Resources

- [Architecture Documentation](docs/developer-guide/architecture.md)
- [API Reference](docs/reference/api/)
- [Plugin Development Guide](docs/developer-guide/plugin-development.md)
- [Roadmap](ROADMAP.md)

## Questions?

If you have questions not covered here:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/ai-research00/AegisPCAP/issues)
3. Ask in [GitHub Discussions](https://github.com/ai-research00/AegisPCAP/discussions)
4. Create a [new issue](https://github.com/ai-research00/AegisPCAP/issues/new/choose)

Thank you for contributing to AegisPCAP! 🎉
