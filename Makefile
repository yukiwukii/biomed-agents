.PHONY: image server test lint help

help:
	@echo "Available targets:"
	@echo "  make server CONFIG=<path>  - Launch the dataset server with the given config file"
	@echo "  make image                 - Build the Docker image for interpreter-env"
	@echo "  make test                  - Run the test suite in parallel"
	@echo "  make lint                  - Run pre-commit hooks and mypy"
	@echo "  make help                  - Show this help message"

server:
	@test -n "$(CONFIG)" || (echo "Error: CONFIG is required. Usage: make server CONFIG=path/to/config.yaml" && exit 1)
	uv run python src/hypotest/dataset_server.py $(CONFIG)

image:
	DOCKER_BUILDKIT=1 docker build -t interpreter-env:latest .

test:
	uv run pytest -n auto

lint:
	uv run prek run -a
	uv run mypy --scripts-are-modules
