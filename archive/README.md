# Smart Real Training System for Forex Bots

## Overview
This project implements a deep reinforcement learning system for training Forex trading bots using genetic algorithms, curriculum learning, and LSTM-based neural networks. It targets high VRAM utilization and includes comprehensive champion analysis.

## Structure
- `src/`: Main source code split into modules (`env.py`, `model.py`, `trainer.py`, `utils.py`, `main.py`)
- `tests/`: Automated tests for each module
- `config.yaml`: Configuration file for parameters
- `logs/`, `checkpoints/`: Output directories for logs and model checkpoints

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Edit `config.yaml` to adjust parameters
3. Run training: `python src/main.py`

## Features
- Modular codebase for maintainability
- Configurable parameters via YAML
- Parallel evaluation for speed
- Automated tests (pytest)
- TensorBoard and logging integration

## Contributing
See `CONTRIBUTING.md` for guidelines.
