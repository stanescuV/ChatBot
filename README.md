# Legal Chatbot with Milvus

A specialized chatbot system designed to handle queries related to the Romanian
Penal Code, utilizing vector embeddings and Milvus for efficient information
retrieval.

## Project Overview

This chatbot is built to process and answer questions about the Romanian Penal
Code. It uses vector embeddings stored in Milvus to provide accurate responses
to legal queries.

## Features

- Vector-based semantic search using Milvus
- Interactive chat interface using Gradio
- Comprehensive testing suite using Ragas
- Dataset processing and management
- Docker support for easy deployment

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Milvus Database
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/stanescuV/ChatBot.git
   cd ChatBot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt 
   ```

   or 
   ``` 
   uv sync
   ```

   it depends on what you usually use

3. Create .env and add API keys inside :

   (I run everything with OpenAI API, didn't try with another provider for the moment)

   ```
   CHATBOT_MODEL_GENERATIVE=


   CHATBOT_MODEL_EMBEDDING=


   MILVUS_IP=localhost

   
   LANGFUSE_SECRET_KEY=
   LANGFUSE_PUBLIC_KEY=
   LANGFUSE_HOST=https://cloud.langfuse.com

   ```

4. Start Milvus using Docker:

   ```bash
   docker-compose up -d
   ```

5. Insert data in Milvus
   ```bash
   python insert_data_milvus.py
   ```

## Usage

1. Start the chatbot

   ```bash
   python -m jupyterlab  
   ```

    ```bash
   uv run jupyter lab  
   ```

2. Open the notebook

   Navigate to `test_chatbot.ipynb`.

3. Run the notebook

   Press ▶️ Play to execute the cells.


## Test API Locally

```python
python server.py 
```

```
uv run server.py 
```

## Data Processing

The project includes Jupyter notebooks for data processing:

- `dataset/refactoring_cod_penal.ipynb`: Processes the Romanian Penal Code
- `test_dataset/ragas_dataset_format.ipynb`: Prepares data for Ragas evaluation



Created and maintained by Stanescu Victor
