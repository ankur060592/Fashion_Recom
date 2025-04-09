# Makefile for managing GenAI Fashion Recommendation Project

# ========== CONFIG ==========
ENV_NAME = genai_env
PYTHON_VERSION = 3.10
REQUIREMENTS_FILE = requirements.txt
DEV_REQUIREMENTS_FILE = dev-requirements.txt

DOCKER_IMAGE = fashion-recom
DOCKER_CONTAINER = fashion-recom-app

# ========== ENV SETUP ==========
.PHONY: setup install clean

fmt:
	@echo "Formatting code with black..."
	pre-commit run --all-files

setup:
	@echo "Creating conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Please activate environment with:"
	@echo "  conda activate $(ENV_NAME)"
	@echo "Next, run 'make install' to install project dependencies."

install:
	@echo "Installing dependencies from $(REQUIREMENTS_FILE)..."
	pip install -r $(REQUIREMENTS_FILE)
	@echo "Installation complete."

install-dev:
	@echo "Installing development dependencies from $(DEV_REQUIREMENTS_FILE)..."
	pip install -r $(DEV_REQUIREMENTS_FILE)
	@echo "Development installation complete."

clean:
	@echo "Removing conda environment '$(ENV_NAME)'..."
	conda remove --name $(ENV_NAME) --all -y
	@echo "Conda environment removed."

# ========== CLEAN FILES ==========
.PHONY: clear-cache

clear-cache:
	@echo "Clearing __pycache__, temp, and output files..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf temp/*
	rm -rf output/*
	@echo "All generated files cleared."

# ========== DOCKER ==========

.PHONY: docker-build docker-run docker-stop

docker-build:
	@echo "Building Docker image '$(DOCKER_IMAGE)'..."
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	@echo "Running Docker container '$(DOCKER_CONTAINER)'..."
	docker run --env-file .env -v C:/Work/GenAI/Fashion_Recom/runs/detect/train7/weights/best.pt:/app/runs/detect/train7/weights/best.pt -p 8501:8501 --name $(DOCKER_CONTAINER) $(DOCKER_IMAGE)

docker-stop:
	@echo "Stopping and removing Docker container '$(DOCKER_CONTAINER)'..."
	docker stop $(DOCKER_CONTAINER) && docker rm $(DOCKER_CONTAINER)

docker-clean:
	@echo "Removing Docker image '$(DOCKER_IMAGE)'..."
	docker rmi $(DOCKER_IMAGE) || true

