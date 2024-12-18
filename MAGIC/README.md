# **Fact-Checking System**

This repository implements a comprehensive fact-checking system that processes claims, retrieves relevant evidence, and classifies the truthfulness of those claims using various AI models and techniques.

## **Table of Contents**

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Prediction](#prediction)
- [Models](#models)
- [Acknowledgements](#acknowledgements)

---

## **Overview**

The fact-checking system combines modern AI tools and methodologies to perform the following tasks:
- **Data Preparation**: Prepares and formats claims and evidence datasets.
- **Evidence Retrieval**: Extracts relevant evidence using Dense Passage Retrieval (DPR).
- **Classification**: Classifies claims based on retrieved evidence using various language models.
- **Evaluation**: Measures accuracy and other metrics for system performance.

Key technologies include:
- **Transformer-based models** for claim classification.
- **Haystack** for dense passage retrieval.
- **LLMs** like GPT-3.5 Turbo and Vicuna-7B for reasoning and classification.

---

## **Directory Structure**

```
.
├── LLM
│   ├── get_chatgpt_prediction.py     # Script to get GPT-based predictions
│   ├── get_chatgpt_arg_prediction.py # Script for argument-based GPT predictions
│   ├── get_vicuna_prediction.py      # Script to get predictions using Vicuna
│   └── ...
├── Retrieval
│   ├── retrieve.py                   # Script for evidence retrieval
│   ├── train_retriever.ipynb         # Notebook to train the retriever model
│   ├── prepare_data.ipynb            # Notebook for data preprocessing
│   └── ...
├── Reader
│   ├── predict.ipynb                 # Notebook for claim prediction
│   ├── train_model.ipynb             # Notebook for training classification models
│   └── ...
├── argument
│   ├── gen_argument.ipynb            # Notebook for generating argument
│   ├── use_chatgpt.py                # Script for generating argument based on chatGPT
│   └── ...
├── dataset                           # Directory containing datasets
├── saved_models                      # Directory for saving trained models
└── retrieval                         # Directory for retrieval outputs
```

---

## **Requirements**

Ensure the following libraries are installed in your environment:

- `torch`
- `transformers`
- `haystack`
- `requests`
- `json`
- `pathlib`
- `numpy`
- `evaluate`
- `scikit-learn`

To install all dependencies, use:

```bash
pip install torch transformers haystack requests numpy evaluate scikit-learn
```

---

## **Usage**

### **1. Data Preparation**
Prepare datasets with the following structure:
- **Claims**: Statements to classify.
- **Evidence**: Textual data supporting or contradicting the claims.

Use scripts in the `Retrieval` directory to preprocess the data and format it for the system.

### **2. Training**
Train retrieval and classification models:
- **Retriever**: Train a Dense Passage Retriever using `train_retriever.ipynb`.
- **Classifier**: Train classification models with `train_model.ipynb`.

### **3. Prediction**
Generate predictions using pretrained or fine-tuned models:
- Use the scripts in the `LLM` directory to classify claims.
- Adjust parameters such as input data paths, model configurations, and output directories as needed.

---

## **Models**

The system supports multiple models for classification and evidence retrieval:

1. **GPT-3.5 Turbo**:
   - Utilized for reasoning and claim classification tasks.
   - Predictions generated using OpenAI's API.

2. **Vicuna-7B**:
   - Fine-tuned LLM for natural language understanding and classification tasks.

3. **Dense Passage Retriever (DPR)**:
   - Used for evidence retrieval to support claims.

---

## **Acknowledgements**

This project leverages open-source tools and pre-trained models, including:
- **OpenAI's GPT models** for robust classification tasks.
- **Haystack** for advanced document retrieval.
- **PyTorch** and **Transformers** for building and fine-tuning custom models.

--- 