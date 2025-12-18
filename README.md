# RAG Pipeline Tutorial

A beginner-friendly tutorial for creating your first Question-Answering (QA) pipeline using Retrieval-Augmented Generation (RAG) with [Haystack](https://haystack.deepset.ai/).

## Overview

This repository contains a Jupyter notebook that demonstrates how to build a generative question-answering system using the RAG approach. The pipeline retrieves relevant documents and uses them as context for generating accurate answers to user questions.

The tutorial uses Wikipedia pages about the [Seven Wonders of the Ancient World](https://en.wikipedia.org/wiki/Wonders_of_the_World) as the document source, but you can easily adapt it to work with your own documents.

## Components Used

- **InMemoryDocumentStore** - Stores documents and their embeddings
- **SentenceTransformersDocumentEmbedder** - Creates embeddings for documents
- **SentenceTransformersTextEmbedder** - Creates embeddings for user queries
- **InMemoryEmbeddingRetriever** - Retrieves relevant documents based on query similarity
- **ChatPromptBuilder** - Builds prompts from templates
- **OpenAIChatGenerator** - Generates responses using OpenAI's GPT models

## Prerequisites

- Python 3.9+
- An [OpenAI API Key](https://platform.openai.com/api-keys)
- (Optional) GPU for faster embedding generation

## Installation

Install the required packages using pip:

```bash
pip install haystack-ai
pip install "datasets>=2.6.1"
pip install "sentence-transformers>=4.1.0"
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/jefferzyn/RAG-Pipeline.git
   cd RAG-Pipeline
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook 27_First_RAG_Pipeline.ipynb
   ```

4. Run the cells sequentially to:
   - Initialize the document store
   - Fetch and embed the Seven Wonders dataset
   - Build the RAG pipeline
   - Ask questions about the ancient wonders

## Example Questions

The notebook includes several example questions you can try:

- "Where is the Hanging Gardens of Babylon?"
- "Why did people build Great Pyramid of Giza?"
- "What does Rhodes Statue look like?"
- "Why did people visit the Temple of Artemis?"
- "What is the importance of Colossus of Rhodes?"
- "What happened to the Tomb of Mausolus?"
- "How did Colossus of Rhodes collapse?"

## Running in Google Colab

This notebook can also be run in Google Colab. For optimal performance, [enable GPU runtime](https://docs.haystack.deepset.ai/docs/enabling-gpu-acceleration).

## Learn More

- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [Filtering Documents with Metadata](https://haystack.deepset.ai/tutorials/31_metadata_filtering)
- [Preprocessing Different File Types](https://haystack.deepset.ai/tutorials/30_file_type_preprocessing_index_pipeline)
- [Creating a Hybrid Retrieval Pipeline](https://haystack.deepset.ai/tutorials/33_hybrid_retrieval)

## Community

- [Subscribe to Haystack Newsletter](https://landing.deepset.ai/haystack-community-updates)
- [Join Haystack Discord](https://discord.gg/haystack)

## License

This tutorial is based on the [Haystack tutorials](https://haystack.deepset.ai/tutorials) by deepset.
