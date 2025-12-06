<div style="text-align: center; display: flex; align-items: center; justify-content: center; background-color: white; padding: 20px; border-radius: 30px;">
  <img src="./static/ASC.jpg" alt="AgentSociety Challenge Logo" width="100" style="margin-right: 20px; border-radius: 10%;">
  <h1 style="color: black; margin: 0; font-size: 2em;">WWW'25 AgentSociety Challenge: WebSocietySimulator</h1>
</div>

# 🚀 AgentSociety Challenge
![License](https://img.shields.io/badge/license-MIT-green) &ensp;
[![Competition Link](https://img.shields.io/badge/competition-link-orange)](https://www.codabench.org/competitions/4574/) &ensp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.18754-b31b1b.svg)](https://arxiv.org/abs/2502.18754)

This repository contains our custome LLM-based recommendation agent built for the **WWW'25 AgentSociety Challenge**. Our work focuses exclusively on the **Recommendation Track**, where the goal is to design agents that can generate high-quality, context-aware item recommendations in a simulated environment based on open-source datasets. We have 2 different implementations of the recommendation agents, see **CS245_RecAgent.py** and **CS245_RecAgent2.py**.

---

## Directory Structure

### 1. **`websocietysimulator/`**  
This is the core library containing all source code required for the competition.

- **`agents/`**: Contains base agent classes (`SimulationAgent`, `RecommendationAgent`) and their abstractions. Participants must extend these classes for their implementations.
- **`task/`**: Defines task structures for each track (`SimulationTask`, `RecommendationTask`).
- **`llm/`**: Contains base LLM client classes (`DeepseekLLM`, `OpenAILLM`).
- **`tools/`**: Includes utility tools:
  - `InteractionTool`: A utility for interacting with the Yelp dataset during simulations.
  - `EvaluationTool`: Provides comprehensive metrics for both recommendation (HR@1/3/5) and simulation tasks (RMSE, sentiment analysis).
- **`simulator.py`**: The main simulation framework, which handles task and groundtruth setting, evaluation and agent execution.

### 2. **`example/`**  
Contains usage examples of the `websocietysimulator` library. Includes sample agents and scripts to demonstrate how to load scenarios, set agents, and evaluate them.

### 3. **`data_process.py`**  
A script to process the raw Yelp dataset into the required format for use with the `websocietysimulator` library. This script ensures the dataset is cleaned and structured correctly for simulations.

### 4. **`CS245_RecAgent.py`** 
Main script to run our recommendation agent variant 1.

### 5. **`CS245_RecAgent2.py`** 
Main script to run our recommendation agent variant 2.

---

## Quick Start

### 1. Install the Library

The repository is organized using [Python Poetry](https://python-poetry.org/). Follow these steps to install the library:

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd websocietysimulator
   ```

2. Install dependencies:
  - Option 1: Install dependencies using Poetry: (Recommended)
    ```bash
    poetry install  && \
    poetry shell
    ```
  - Option 2: Install dependencies using pip(COMING SOON):
    ```bash
    pip install websocietysimulator
    ```
  - Option 3: Install dependencies using conda:
    ```bash
    conda create -n websocietysimulator python=3.11 && \
    conda activate websocietysimulator && \
    pip install -r requirements.txt && \
    pip install .
    ```

3. Verify the installation:
   ```python
   import websocietysimulator
   ```

---

### 2. Data Preparation

1. Download the raw dataset from the Yelp[1], Amazon[2] or Goodreads[3].
2. Run the `data_process.py` script to process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
   ```
- Check out the [Data Preparation Guide](./tutorials/data_preparation.md) for more information.
- **NOTICE: You Need at least 16GB RAM to process the dataset.**

---

### 3. Organize Your Data

Ensure the dataset is organized in a directory structure similar to this:

```
<your_dataset_directory>/
├── item.json
├── review.json
├── user.json
```

You can name the dataset directory whatever you prefer (e.g., `dataset/`).

---
### 4. Run our Recommendation Agent

Setup configurations in ```run_experiments()``` in ```CS245_RecAgent.py```. The parameters are:  

**num_tasks**: number of examples used for simulation  
**model**: the backend LLM  
**num_candidate_plans**: number of plans the agent generates  
**use_plan**: whether to use the planning  
**extra_thinking_tokens**: whether to use extra thinking tokens  
**memory**: whether to activate memory module  

**Usage**: ```python CS245_RecAgent.py```

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

[1] Yelp Dataset: https://www.yelp.com/dataset

[2] Amazon Dataset: https://amazon-reviews-2023.github.io/

[3] Goodreads Dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
