# Legal Chatbot with Milvus

A specialized chatbot system designed to handle queries related to the Romanian Penal Code, utilizing vector embeddings and Milvus for efficient information retrieval.

## Project Overview

This chatbot is built to process and answer questions about the Romanian Penal Code. It uses vector embeddings stored in Milvus to provide accurate responses to legal queries.

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

3. Start Milvus using Docker:
   ```bash
   docker-compose up -d
   ```

4. Insert data in Milvus 
   ```bash
   python insert_data_milvus.py
   ```

## Usage

1. Start the chatbot:
   ```bash
   python chatbot.py
   ```

2. The Gradio interface will be available at `http://localhost:7860`


## Data Processing

The project includes Jupyter notebooks for data processing:
- `dataset/refactoring_cod_penal.ipynb`: Processes the Romanian Penal Code
- `test_dataset/ragas_dataset_format.ipynb`: Prepares data for Ragas evaluation

## Contact

STANESCU VICTOR : stanescuvictor12@gmail.com

---
Created and maintained by Stanescu Victor