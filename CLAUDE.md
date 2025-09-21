# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "overthink-black", a research project that uses genetic algorithms to optimize mathematical problem generation to maximize reasoning token usage in language models. The system evaluates model responses by counting reasoning tokens (particularly for models like deepseek-reasoner) to find problem formulations that trigger more extensive reasoning.

## Core Architecture

### Main Components

- **main.py**: Contains the core genetic algorithm implementation
  - `MathProblem` class: Represents individual math problems with premises and questions
  - `GeneticAlgorithm` class: Implements the evolutionary optimization process
  - Main execution loop that loads data, evolves problems, and tracks results

- **model.py**: LLM interaction layer
  - `TargetModel`: Handles synchronous and asynchronous calls to target models for evaluation
  - `AttackerModel`: Specialized for problem decomposition and preprocessing
  - Supports both OpenAI-compatible APIs and custom endpoints

- **utils.py**: Data handling utilities
  - `read_dataset()`: Loads JSONL datasets from the data/ directory  
  - `random_sample()`: Random sampling utility for dataset selection

- **prompts.py**: Prompt templates
  - `SPLIT_QUESTION`: Template for decomposing problems into premises and questions

### Data Structure

- **data/**: Contains training and test datasets
  - `train.jsonl`, `test.jsonl`: Main datasets
  - `math_test.jsonl`, `svamp_test.json`: Additional test datasets
  - Datasets contain mathematical word problems for optimization

### Experiment Tracking

- Uses SwanLab for experiment logging and visualization
- Tracks fitness evolution, model performance, and hyperparameters
- Logs stored in `swanlog/` directory

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses Chinese PyPI mirror)
uv sync

# Activate virtual environment if needed
source .venv/bin/activate
```

### Running the System
```bash
# Run the main genetic algorithm optimization
python main.py
```

### Configuration

- **Environment variables**: Configure in `.env`
  - `BASE_URL`: API endpoint for LLM services
  - `API_KEY`: Authentication key for LLM API

- **Algorithm parameters**: Modify in `main()` function of main.py
  - `population_size`: Number of problems in each generation
  - `elite_rate`: Fraction of top performers to preserve
  - `crossover_rate`: Probability of genetic crossover
  - `mutation_rate`: Probability of mutations
  - `generations`: Number of evolutionary iterations

## Key Design Patterns

- **Fitness Function**: Measures reasoning token count from model responses
- **Genetic Operations**: 
  - Crossover exchanges premises or questions between problems
  - Mutation adds/removes premises from the problem pool
  - Elite selection preserves best-performing problems
- **Async Processing**: Batch evaluation of multiple problems for efficiency
- **Problem Representation**: Structured as premises list + question string