# Email Generator with Langchain, FAISS, and Streamlit
# dataset link-https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

This project demonstrates the creation of an Email Generator using the following technologies:

- **Langchain**: A framework that helps in building applications with language models.
- **FAISS**: A library for efficient similarity search, which helps in finding the most relevant context or content for email generation.
- **Sentence Transformers**: Used to create embeddings for text, allowing us to represent the emails' content in vector space for similarity search.
- **Streamlit**: A simple and fast way to build the front end for the email generator app, providing an interactive user interface.

## Features

- **Similarity Search**: Uses FAISS for vector-based similarity search to find the most relevant emails based on the input.
- **Text Embedding**: Leverages Sentence Transformers to convert input text into embeddings for comparison.
- **Email Generation**: Generates a personalized email based on the retrieved context and user input using Langchain.
