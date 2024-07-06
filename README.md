# Kaggle GPU Instances to MongoDB for GSM8K Response Generation

This repository facilitates the use of multiple free Kaggle 2xT4 GPU instances connected to a central MongoDB database (free tier) to generate and refine responses to tasks such as GSM8K math questions. The generated responses can be used to create synthetic training data or benchmark a model on GSM8K.

## Features

- **Parallel Model Execution**: Utilizes two GPUs per instance to run two copies of the model in parallel, maximizing data generation efficiency.
- **Batch Processing**: Processes data in batches to further enhance speed.
- **Scalable Architecture**: Connects multiple Kaggle instances to a central MongoDB database to increase the speed of data generation.

## Usage

### File Descriptions

- **one_shot_prompt.txt**: A text file used to extract the answer to a math question as a number for evaluation.
- **retry_prompt.txt**: A prompt file to instruct the model to review its previous answer (if incorrect) and attempt to answer again.
- **requirements.txt**: Lists necessary libraries, most of which are pre-installed on Kaggle instances.

### Main Script Capabilities

- **MongoDB Table and Collection Creation**: Creates tables and collections with the GSM8K train and test sets for response generation and evaluation.
- **Retry Table Creation**: Generates a "retry" table for refining and making second attempts at answering.
- **PEFT Model Integration**: Wraps the base model in a PEFT model to evaluate and generate responses with fine-tuned models.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script**:
   ```bash
   python main_script.py
   ```

## Future Development

This framework is an ongoing project and will be continuously developed to include more use cases. There are numerous methods to generate responses and prompt an LLM (Large Language Model) to refine, reattempt, and explore questions or tasks. Contributions and suggestions are welcome!

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please open an issue or contact us directly.

Happy coding!
```

Feel free to further customize the README to better fit your project's specifics or add additional sections as needed.
