# Schelling Segregation Model Simulation

This project is a Python implementation of the Schelling Segregation Model, a classic agent-based model that simulates how local preferences for similar neighbors can lead to large-scale segregation patterns in a population. The implementation uses **NetworkX** to represent the spatial grid and **Matplotlib** for visualizing the simulation.

## Overview

The Schelling model explores how individual agents' preferences for living near others of their own type can lead to the emergence of global segregation patterns, even when agents are tolerant of some diversity in their neighborhood. In this simulation, agents are distributed on a 2D grid with a mixture of empty spaces, and each agent evaluates its satisfaction based on the composition of its neighborhood.

### Key Features

- **Agent-Based Model:** Agents are placed on a 2D lattice grid where they interact with their 8 neighboring cells.
- **Dynamic Movement:** Unsatisfied agents (based on tolerance thresholds) attempt to move to empty cells, while satisfied agents remain in place.
- **Multiple Classes:** Agents are divided into two or more classes, representing distinct types or groups in the population.
- **Adaptive Parameters:** The simulation allows for varying tolerance thresholds and empty ratios to explore different segregation dynamics.
- **Visualization:** The evolution of the grid over time is animated using Matplotlib, with color coding for different agent types, empty spaces, and stuck agents (agents that cannot move).

## Installation

To get started with the project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/henriqueccmac/SchellingModel.git
cd schelling-model-simulation
pip install networkx matplotlib numpy
```

## Dependencies

- Python 3.6+
- NetworkX
- Matplotlib
- NumPy

## Running the simulation

The main code is contained in the **schelling.py** file. To run the simulation, execute the following command:

```bash
    python schelling.py
```

### Parameters

- **Grid Size:** The size of the 2D grid (e.g., 50x50).
- **Empty Ratio:** The proportion of the grid that remains empty, allowing agents to move around. (e.g., 0.5 for 50% empty spaces).
- **Tolerance Threshold:** A value between 0.0 and 1.0 representing the proportion of similar neighbors an agent requires to be satisfied. Lower values indicate higher tolerance for diversity, while higher values lead to more segregation.
- **Agent Classes:** The number of distinct agent types in the simulation (typically 2 for binary segregation).

### Example Usage

By default, the simulation runs on a `50x50` grid with two agent classes, a `0.5` empty ratio, and a tolerance threshold of `0.6`

### Simulation Logic

- **Grid Construction:**
  - A 2D lattice grid is constructed using **NetworkX**, where diagonal edges are included to simulate agents' interactions with their diagonal neighbors (total 8 neighboring cells).

- **Agent Assignment:**
  - Agents are randomly assigned to positions on the grid, ensuring that a proportion of spaces remain empty, as determined by the `empty_ratio`.

- **Satisfaction Calculation:**
  - Each agent evaluates its satisfaction by counting the number of neighbors of the same class compared to the total number of occupied neighbors. If the satisfaction ratio falls below the agent's **tolerance threshold**, the agent becomes unsatisfied.

- **Agent Movement:**
  - Unsatisfied agents try to move to a nearby empty cell. If no such cell is available, they are considered "stuck" and cannot move further.

- **Simulation Termination:**
  - The simulation continues until all agents are either satisfied or stuck. The system evolves step by step, recalculating satisfaction and relocating unsatisfied agents.

- **Visualization:**
  - The grid is visualized and updated in each generation using **Matplotlib**. Each agent class is assigned a unique color, while empty cells are white, and stuck agents are purple.

### Outputs

After running the simulation, the model provides visual and statistical outputs:

- **Grid Visualization:** Displays the state of the grid with different colors representing different agent classes, empty spaces, and stuck agents.

- **Homogeneity & Moran's I:** Two statistical measures of segregation:
  - **Homogeneity:** Measures the proportion of agents surrounded by similar agents.
  - **Moran's I:** Quantifies spatial autocorrelation (clustering) of similar agents.

Plots for **Homogeneity** and **Moran's I** are generated after each simulation run, illustrating how these metrics evolve with varying tolerance thresholds and empty ratios.
