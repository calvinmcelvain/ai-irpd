# AI-IRPD Project

This project focuses on creating game-theory categories from summaries of instances/cases in an experimental economics game (Indefinitely Repeated Prisoner-Dilemma games (IRPD)) and classifying those summaries into the created categories.

**Note**: This is currently for a working paper. No data or prompts are available at this time. If code is used from this repository, it should cite our paper: "Things Change: Changing Actions and Strategies in Indefinitely Repeated PD Games" (Cooper, Kagel, Qi, McElvain).

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

### Intra-Model (Replication) Tests

- **Description**: Conduct replications of certain stages to measure the stability and replicability of the categorization process.

### Cross-Model Validation Tests

- **Description**: Validate/Analyze certain stage outputs using different Large Language Models (LLMs).

## Project Structure

The project is organized into several modules/packages and scripts to handle different stages and functionalities:

- **/src**
  - **llms/**: Contains the LLM model configurations and request handling, as well as the IRPD testing class.
    - **models.py**: Aggregates all LLMs and their clients.
  - **testing/**: Contains all testing modules and **stages/** package.
- **tools/**: Tools not otherwise used for testing.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact [mcelvain@uiowa.edu].
