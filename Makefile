# Makefile for setting up the GenAI environment

.PHONY: setup install clean

setup:
	conda create --name genai_env python=3.10 -y
	conda activate genai_env
	pip install -r requirements.txt


install:
	pip install -r requirements.txt

clean:
	conda remove --name genai_env --all -y
