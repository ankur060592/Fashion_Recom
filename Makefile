# Makefile for managing GenAI Fashion Recommendation Project

# ========== CONFIG ==========
ENV_NAME = genai_env
PYTHON_VERSION = 3.10
REQUIREMENTS_FILE = requirements.txt
DEV_REQUIREMENTS_FILE = dev-requirements.txt

# ========== ENV SETUP ==========
.PHONY: setup install clean

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

