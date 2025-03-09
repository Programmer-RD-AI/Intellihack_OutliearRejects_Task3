# LLM Fine-tuning Challenge: Enhancing Qwen 2.5-3B for AI Research QA

This project demonstrates a comprehensive approach to fine-tuning the Qwen 2.5-3B model for specialized AI research question-answering. The implementation focuses on creating an efficient domain-specific QA system that can accurately answer questions about technical AI infrastructure concepts, particularly those related to distributed file systems and performance optimization.

## 📥 Model Access

Due to their large file sizes, the trained models are not included in this repository but can be accessed via the following links:

- **Fine-tuned Model**: [Download from Google Drive](https://drive.google.com/drive/folders/1-1ntQq51rkWsvrOXY1HA73zlD8mZh8Kj?usp=sharing)
- **Quantized Model**: [Download from Google Drive](https://drive.google.com/drive/folders/1-vSd2iDNG_9hxZ_H79MVPYO9HYKs8nRh?usp=sharing)
- **Complete Project Files**: [Access on Google Drive](https://drive.google.com/drive/folders/18qAq284GfgvIfAZYQAJiMeMkcELDkEq9?usp=sharing) (where the project was run and tested)

## 📋 Project Overview

The project implements a complete pipeline for:

1. Processing technical research documents
2. Generating high-quality synthetic QA pairs
3. Fine-tuning Qwen 2.5-3B using QLoRA
4. Building a retrieval-augmented generation (RAG) system
5. Evaluating model performance using multiple metrics

## 🧩 Components

### Document Processing

- Extracts structured information from technical markdown documents
- Segments text into meaningful chunks for context preservation
- Handles specialized formatting and technical content

### QA Generation

- Creates synthetic question-answer pairs from processed documents
- Employs instruction templates optimized for technical QA formatting
- Generates training and validation datasets

### Fine-tuning Pipeline

- Implements QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning
- Optimizes hyperparameters for the technical domain
- Uses BitsAndBytes for quantization
- Tracks training with Weights & Biases integration

### RAG System

- FAISS-based vector store for semantic document retrieval
- Optimized embeddings for technical content
- Context-aware question answering

### Evaluation Framework

- Multiple metrics including ROUGE, BLEU, and custom accuracy measures
- Comprehensive evaluation of model output quality

## 🚀 Usage

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA.git
cd LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA

# Install dependencies
uv sync

# Run
uv run llm_fine_tuning_challenge_enhancing_qwen_2_5_3b_for_ai_research_qa.py
```

## 📊 Results

The fine-tuned model demonstrates significant improvements over the base model for technical AI research questions:

- Higher accuracy in addressing complex technical concepts
- Improved response quality for system architecture questions
- Better context maintenance for multi-part technical explanations

## 🧪 Dataset

The model is trained using the Q3 dataset containing detailed technical documentation about:

- Fire-Flyer File System (3FS) architecture
- Chain Replication with Apportioned Queries (CRAQ)
- Performance optimizations for distributed systems
- AI infrastructure components

## 📃 License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
