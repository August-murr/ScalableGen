# Kaggle Multi-GPU Math Question Response Generator

This repository connects multiple Kaggle free 2xT4 GPU instances to a central MongoDB database (free tier) to generate and refine responses for tasks such as GSM8K math questions. The generated responses can be used to create synthetic training data or benchmark models on GSM8K. 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Overview
To maximize efficiency, this framework runs two copies of a model on two GPUs in parallel and processes data in batches. By connecting multiple Kaggle instances, it increases the speed of data generation.

## Features
- **Parallel Model Execution**: Utilizes two GPUs per instance to run models in parallel.
- **Batch Processing**: Processes data in batches for efficiency.
- **Multiple Instances**: Connects multiple Kaggle instances to a central MongoDB for faster data generation.
- **Answer Extraction**: Uses `one_shot_prompt.txt` to extract math question answers as numbers.
- **Answer Refinement**: Uses `retry_prompt.txt` to prompt the model to review and correct previous answers.
- **Model Evaluation**: Supports wrapping base models in PEFT (Parameter-Efficient Fine-Tuning) models for evaluation and response generation.

## Setup
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/kaggle-multi-gpu-math-response.git
    cd kaggle-multi-gpu-math-response
    ```
2. **Install dependencies**:
    Kaggle instances come with most libraries pre-installed. However, you can install any additional requirements with:
    ```sh
    pip install -r requirements.txt
    ```
3. **Setup MongoDB**:
    - Create a MongoDB database and obtain the connection URI.
    - Update the MongoDB connection settings in the script.

## Usage
1. **Run the main script**:
    ```sh
    python main.py
    ```
2. **Generate responses**:
    - The script will generate responses using the models and store them in MongoDB collections.
3. **Refine responses**:
    - The script will use the `retry_prompt.txt` to refine incorrect answers and store them in a "retry" table.

## Modules
- **Data Initialization**:
    - Scripts to create MongoDB tables and collections with GSM8K train and test sets.
- **Response Generation**:
    - Modules to generate responses using base models and PEFT models.
- **Answer Refinement**:
    - Modules to refine and reattempt answers based on model feedback.

## Contributing
Contributions are welcome! Since this is an ongoing project, there are many opportunities to explore different methods for response generation and refinement. Feel free to open issues or submit pull requests.

