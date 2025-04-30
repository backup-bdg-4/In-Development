---
name: repo
type: repo
agent: CodeActAgent
---

# Backdoor AI

This repository contains the code for a web application with a FastAPI backend and a React frontend. The application is an AI-powered platform with robust security features, rate limiting, and input validation.

## General Setup:

To set up the repository:

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Run the application using Docker Compose:
   ```bash
   docker-compose up
   ```

## Repository Structure:

### Backend:
- Located in the `backend/app` directory
- Main application entry point: `backend/app/main.py`
- Configuration: `backend/app/config.py`
- Utilities:
  - Security: `backend/app/utils/security_utils.py` (custom SecureHeaders implementation)
  - API endpoints: `backend/app/api/`
  - Models: `backend/app/models/`
- Testing:
  - Uses pytest for testing

### Frontend:
- Located in the `frontend` directory
- Built with React

### Configuration:
- Environment variables and configuration settings
- Docker Compose for containerization
- Render.yaml for deployment configuration

### CI/CD:
- GitHub Actions workflows for continuous integration and deployment

## Security Features:
- Custom SecureHeaders implementation for HTTP security headers
- Rate limiting with slowapi
- Input validation and sanitization
- Query blocklist for harmful patterns
- Client information tracking
- Trusted host middleware

## Development Guidelines:
1. Always run security checks before deploying:
   ```bash
   # Check for security vulnerabilities in dependencies
   pip-audit
   
   # Run linting
   flake8
   ```

2. Follow secure coding practices:
   - Validate all user inputs
   - Use parameterized queries
   - Apply the principle of least privilege
   - Keep dependencies updated

3. Code style:
   - Follow PEP 8 guidelines
   - Use type hints
   - Document functions and classes with docstrings