# Makefile for setting up the GenAI environment

# Define the environment name
ENV_NAME = genai_env

# Define the Python version
PYTHON_VERSION = 3.10

# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

.PHONY: setup install clean

# Setup the environment
setup:
	@echo "Creating conda environment $(ENV_NAME) with Python $(PYTHON_VERSION)..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Activating conda environment $(ENV_NAME)..."
	# Note: The following line will not work directly in Makefile due to shell limitations.
	# You need to manually activate the environment in your terminal before proceeding.
	# conda activate $(ENV_NAME)
	@echo "Please activate the environment manually using: conda activate $(ENV_NAME)"
	@echo "Installing dependencies from $(REQUIREMENTS_FILE)..."
	# Install torch first to avoid conflicts
	pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121
	pip install -r $(REQUIREMENTS_FILE)
	@echo "Setup complete!"

# Install dependencies
install:
	@echo "Activating conda environment $(ENV_NAME)..."
	# Note: The following line will not work directly in Makefile due to shell limitations.
	# You need to manually activate the environment in your terminal before proceeding.
	# conda activate $(ENV_NAME)
	@echo "Please activate the environment manually using: conda activate $(ENV_NAME)"
	@echo "Installing dependencies from $(REQUIREMENTS_FILE)..."
	pip install -r $(REQUIREMENTS_FILE)
	@echo "Installation complete!"

# Clean the environment
clean:
	@echo "Removing conda environment $(ENV_NAME)..."
	conda remove --name $(ENV_NAME) --all -y
	@echo "Cleanup complete!"
