# Multi-Agent Reinforcement Learning

- Asger Sturis Tang, s184305
- Frederik Møller Sørensen, s184306
- Joachim Pors Andreassen, s184289

## Introduction

This repository contains the code for our master thesis at The Technical University of Denmark ([DTU](https://www.dtu.dk/english/)). The project focuses on multi-agent reinforcement learning, specifically developing pricing and rebalancing strategies for urban mobility in different cities using the Soft Actor-Critic algorithm (SAC) and the Advantage Actor-Critic algorithm (A2C).

## Setup

### Requirements

- [Python](https://www.python.org/downloads/release/python-31011/) 3.10.xx
- [Poetry](https://python-poetry.org/) for dependency management
- [CPLEX 22.1.1](https://community.ibm.com/community/user/ai-datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students?CommunityKey=ab7de0fd-6f43-47a9-8261-33578a231bb7&tab=) - Requires license or a student account!

### Initialise project
Before doing the below, make sure you have completed the appropriate steps to install all of the correct requirements.

1. Clone the repository:
   ```bash
   git clone https://github.com/Jungar111/multi_agent_reinforcement_learning
   ```
2. Navigate to the cloned directory:
   ```bash
   cd multi_agent_reinforcement_learning
   ```
3. Get the latest .lock file using Poetry:
   ```bash
   poetry lock
   ```
4. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

The project includes two main scripts:

- `main.py`: Runs the A2C (Advantage Actor-Critic) algorithm.
- `main_SAC.py`: Executes the Soft Actor-Critic (SAC) algorithm.

To run the A2C algorithm for San Francisco, use:

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
   - Note that the `saved_files` requires the following sub-folder architecture, which is not inherited from the repo:
   - `ckpt`
      - `scenario_{city}`
   - `cplex_logs`
      - `matching`
         - `scenario_{city}` 
      - `rebalancing`
         - `scenario_{city}`


## Acknowledgments

This work was conducted as part of a master thesis at DTU. We would like to thank Francisco Camara Pereira, Filipe Rodrigues, Carolin Samanta Schmidt and DTU for their support, guidance and endless fruitful discussions.
