# Multi-Agent Reinforcement Learning

- Asger Sturis Tang, s184305
- Frederik Møller Sørensen, s184306
- Joachim Pors Andreassen, s184289

## Introduction

This repository contains the code for our master thesis at Danmarks Tekniske Universitet (DTU). The project focuses on
multi-agent reinforcement learning, specifically developing pricing and rebalancing strategies for urban mobility in
different cities using SAC and A2C.

## Setup

### Requirements

- Python 3.x
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jungar111/multi_agent_reinforcement_learning
   ```
2. Navigate to the cloned directory:
   ```bash
   cd multi_agent_reinforcement_learning
   ```
3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

The project includes two main scripts:

- `main.py`: Runs the A2C (Advantage Actor-Critic) algorithm.
- `main_SAC.py`: Executes the Soft Actor-Critic (SAC) algorithm.

To run the default simulation for San Francisco, use:

```bash
python main.py
```

Or for the SAC algorithm:

```bash
python main_SAC.py
```

### Customization

You can customize the city for simulation by modifying the data source in the script. The project defaults to San Francisco but supports other cities included in the `data` folder.

## Project Structure

- `data`: Contains city-specific data for simulations.
- `images`: Stores images and visual assets.
- `multi_agent_reinforcement_learning`: Main module containing:
  - `algos`: Algorithm implementations.
  - `build`: Compiled files.
  - `cplex_mod`: CPLEX model files.
  - `data_models`: Data models for the project.
  - `envs`: Environment configurations for the RL agents.
  - `evaluation`: Evaluation scripts and utilities.
  - `misc`: Miscellaneous scripts and files.
  - `plots`: Code for generating plots.
  - `saved_files`: Saved checkpoints and logs.
  - `utils`: Utility scripts and helpers.
- `notebooks`: Jupyter notebooks for exploratory data analysis and visualizations.
- `saved_files`: Contains RL logs, CPLEX logs, and checkpoints.


## Acknowledgments

This work was conducted as part of a master thesis at DTU. We would like to thank Francisco Camara Pereira, Filipe Rodrigues and Carolin Samanta Schmidt, and DTU for their support and guidance.
