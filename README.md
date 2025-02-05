# AI-IRPD Project

This project is a specific research project aimed at creating game-theory categories from summaries of instances in an experimental economics game (Indefinitely Repeated Prisoner-Dilemma games (IRPD)), and classifying those summaries into the created categories. This project is not generally meant for public use due to its specialized nature.

## Overview

The research project involves several stages to process and classify summaries derived from experimental data. The stages are as follows:

### Stage 0: Summary Creation

- **Description**: Create summaries from raw experimental data.
- **Note**: This stage is not currently set up; we are using summaries created by other research assistants.

### Stage 1: Category Creation

- **Description**: Create categories from summaries of game instances.
- **Instance Types**: Instances are usually broken into two parts, denoted as instance types, typically called `ucoop` or `udef`, but sometimes `coop` or `def` depending on the instance.

### Stage 1r: Category Refinement

- **Description**: Refine the categories created in Stage 1 by merging overlapping categories.

### Stage 1c: Unified Category Creation

- **Description**: Combine Stage 1 and Stage 1r but unify the instance types into one, creating a unified category set. Remove instance type-specific categories from Stage 1r that have a name with a cosine similarity score greater than or equal to 0.3 (compared to a category in the unified set).

### Stage 2: Summary Classification

- **Description**: Classify individual summaries into the created categories.

### Stage 3: Classification Ranking

- **Description**: Rank the classifications for each summary from Stage 2.

## Additional Tests

### Replication/Stability Tests

- **Description**: Conduct replications of certain stages to measure the stability and replicability of the categorization process.

### Cross-Model Validation Tests

- **Description**: Validate/Analyze certain stage outputs using different Large Language Models (LLMs).

## Project Structure

The project is organized into several modules and scripts to handle different stages and functionalities:

- **models/**: Contains the LLM model configurations and request handling, as well as the IRPD testing class.
- **schemas/**: Defines the JSON schemas for different stages.
- **utils/**: Utility functions for data preparation, replication analysis, and stage processing.
  - **plots/**: Plots for visualizing replication/stability tests.

## Getting Started

### Installation

1. Clone the repository:

```sh
git clone https://github.com/calvinmcelvain/ai-irpd.git
```

2. Install the required packages:


```sh
pip install -r requirements.txt
```

3. Set up environment variables:
    - Create a `configs.env` file in the root directory with the necessary configurations (e.g., LLM API keys and/or main preoject directory).

## Contributing

This project is primarily for internal research purposes. If you have any suggestions or improvements, please feel free to reach out.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact [mcelvain@uiowa.edu].
