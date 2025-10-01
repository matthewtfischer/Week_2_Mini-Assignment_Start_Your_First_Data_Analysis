PYTHON = python

.PHONY: lint format check-format clean

# Run linting with flake8
lint:
	flake8 .

# Auto-format code with black + isort
format:
	black .
	isort .

# Check formatting without changing files
check-format:
	black --check .
	isort --check-only .

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
