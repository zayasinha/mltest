# Contributing to ML Project Dashboard

Thank you for your interest in contributing! This document provides guidelines for contributing to this ML project.

## How to Contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** and test them
5. **Commit your changes**: `git commit -m "Add your message"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request** on GitHub

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ml-project-dashboard.git
cd ml-project-dashboard

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest  # if you add tests

# Run the application
python app.py

# Run the dashboard
python run_dashboard.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions small and focused

## Adding New Features

- Discuss major features in Issues before implementing
- Add tests for new functionality
- Update documentation as needed
- Ensure backward compatibility

## Reporting Issues

- Use GitHub Issues to report bugs
- Include steps to reproduce the issue
- Provide system information and error messages
- Suggest potential solutions if possible

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.