# AIDE ML Tutorial

This tutorial explains the structure and usage of the `aideml` package, a tool for automated machine learning engineering using agentic tree search.

## Overview

AIDE ML (AI-Driven Exploration) uses Large Language Models (LLMs) to iteratively generate, debug, and improve machine learning python scripts. It maintains a "solution tree" where each node is a potential solution (code), and it navigates this tree to maximize a defined metric.

## 1. Installation

The package can be installed directly from PyPI:

```bash
pip install -U aideml
```

Or from source for development:

```bash
git clone https://github.com/WecoAI/aideml.git
cd aideml
pip install -e .
```

Prerequisites:
- Python 3.10+
- An API key for your LLM provider (e.g., `OPENAI_API_KEY`).

## 2. Directory Structure for Tasks

Before running AIDE, you need a directory containing your dataset. For example:

```
my_task/
├── train.csv
├── test.csv
└── metadata.txt (optional)
```

## 3. Usage via CLI

The simplest way to use AIDE is globally via the command line.

### Basic Command

```bash
aide data_dir="path/to/my_task" \
     goal="Predict the target column value" \
     eval="Accuracy"
```

### Common Options

You can override any configuration parameter using the dot notation:

```bash
aide data_dir="example_tasks/house_prices" \
     goal="Predict house prices" \
     eval="RMSE" \
     agent.steps=20 \
     agent.code.model="gpt-4o"
```

## 4. Usage via Python API

For integration into other Python projects, you can use the `aide` class interfaces directly.

### The `Experiment` Class

The main entry point is the `Experiment` class in `aide/__init__.py`.

```python
import aide
import logging

# Setup logging to see progress
logging.basicConfig(level=logging.INFO)

# Define the experiment
exp = aide.Experiment(
    data_dir="example_tasks/house_prices",
    goal="Predict the sales price for each house",
    eval="RMSE (Root Mean Squared Error)"
)

# Run the agent for a specific number of steps
# Returns a Solution object containing the best code and metric
best_solution = exp.run(steps=10)

print(f"Best Metric: {best_solution.valid_metric}")
print("Best Code:")
print(best_solution.code)
```

### Customizing Configuration in Python

The `Experiment` class loads default configurations. You can inspect or modify `exp.cfg` before running, although passing arguments to the constructor or using the CLI approach is standard.

## 5. Configuration Guide

AIDE uses `OmegaConf` for configuration. The defaults are defined in `aide/utils/config.yaml`.

### Key Sections:

1.  **General**:
    *   `data_dir`: Path to data.
    *   `goal`: Text description of the task.
    *   `eval`: Text description of the evaluation metric.
    *   `workspace_dir`: Where code execution happens (default: `workspaces`).
    *   `log_dir`: Where results are saved (default: `logs`).

2.  **Agent (`agent`)**:
    *   `steps`: Total numbering of search iterations (default: 20).
    *   `k_fold_validation`: Number of folds for CV (default: 5).
    *   `data_preview`: Give the LLM a snippet of the data (default: True).

3.  **Models (`agent.code` / `agent.feedback`)**:
    *   `model`: Model name (e.g., `o3-mini`, `claude-3-5-sonnet`).
    *   `temp`: Temperature for sampling.

4.  **Search (`agent.search`)**:
    *   `num_drafts`: Number of initial independent attempts.
    *   `debug_prob`: Probability of trying to fix a buggy node vs. drafting/improving.
    *   `max_debug_depth`: How deep the debugging chain can go.

## 6. Output & Visualization

After a run, check the `logs/<experiment_id>` directory.
- **`best_solution.py`**: The final, best-performing script.
- **`tree_plot.html`**: An interactive visualization of the search tree.
- **`journal.json`**: The full raw history of the session.

To visualize the tree: Open `logs/<id>/tree_plot.html` in your browser.

## 7. Web UI

A built-in Streamlit UI allows for interactive experimentation.

```bash
# From the repository root
streamlit run aide/webui/app.py
```

This interface wraps the core `Experiment` logic and visualizes the journal updates in real-time.
