import os
import re
from pathlib import Path
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoModel,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
import wandb
import subprocess
import shutil
import argparse
from typing import List, Optional, Dict, Any
import time
from llama_cpp import Llama
import faiss
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import gc
import nltk
import json

nltk.download("punkt_tab")
# Set this environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear cache at startup
gc.collect()
torch.cuda.empty_cache()

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # or "Qwen/Qwen2.5-3B"
DATA_DIR = "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data"
OUTPUT_DIR = "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/fine_tuned_model"
MAX_LENGTH = 2048
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA configuration
LORA_CONFIG = LoraConfig(
    r=32,  # rank
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


class DocumentProcessor:
    def __init__(self, documents_dir):
        self.documents_dir = Path(documents_dir)
        self.documents = {}
        self.load_documents()

    def load_documents(self):
        """Load all markdown documents from the specified directory."""
        for file_path in self.documents_dir.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract title from filename or first heading in the document
                title = file_path.stem
                self.documents[title] = content
        print(f"Loaded {len(self.documents)} documents.")

    def chunk_documents(self, chunk_size=1500, overlap=150):
        """Chunk documents into smaller pieces with overlap."""
        chunked_docs = []

        for title, content in self.documents.items():
            # Remove markdown formatting for cleaner text
            text = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            text = re.sub(r"#+ ", "", text)
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

            # Split into sentences (rough approximation)
            sentences = re.split(r"(?<=[.!?])\s+", text)

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    # Start new chunk with overlap
                    overlap_tokens = (
                        current_chunk[-overlap:]
                        if overlap < len(current_chunk)
                        else current_chunk
                    )
                    current_chunk = overlap_tokens + [sentence]
                    current_length = len(current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    {
                        "title": title,
                        "chunk_id": i,
                        "text": chunk.strip(),
                    }
                )

        return chunked_docs


class QAGenerator:
    def __init__(self, model_name="Qwen/Qwen1.5-7B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate_qa_pairs(self, chunks, num_questions_per_chunk=3):
        qa_pairs = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}")

            # Better prompt with examples to avoid placeholders
            prompt = f"""Given the text below from an AI research paper, generate {num_questions_per_chunk} detailed question-answer pairs.

    TEXT:
    {chunk["text"]}

    INSTRUCTIONS:
    - Create substantive, specific questions about key concepts in the text
    - Write comprehensive answers using information directly from the text
    - DO NOT generate generic or placeholder questions
    - DO NOT use phrases like "write a question here" or "comprehensive answer here"
    - Use this exact format for each pair:

    Q1: What is [specific concept from text]?
    A1: [Detailed answer explaining the concept based on the text]

    Here are {num_questions_per_chunk} question-answer pairs about this text:
    """
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Debug output
                print(f"\nModel response for chunk {i + 1}:")
                print(response[-200:])  # Show the last 200 characters

                # Extract questions and answers
                pairs = self.extract_qa_pairs(response)

                print(f"Extracted {len(pairs)} QA pairs from chunk {i + 1}")

                if len(pairs) == 0:
                    # Fallback prompt with even more explicit instructions
                    fallback_prompt = f"""I need exactly {num_questions_per_chunk} question-answer pairs about this AI research text.

    TEXT:
    {chunk["text"]}

    FORMAT YOUR RESPONSE LIKE THIS - with real content, not placeholders:
    Q1: [Real specific question about the content]
    A1: [Real detailed answer from the content]

    Q2: [Real specific question about the content]
    A2: [Real detailed answer from the content]

    Q3: [Real specific question about the content]
    A3: [Real detailed answer from the content]
    """
                    inputs = self.tokenizer(fallback_prompt, return_tensors="pt").to(
                        self.model.device
                    )

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.8,  # Slightly higher to encourage creativity
                        top_p=0.92,
                        repetition_penalty=1.3,
                        do_sample=True,
                    )

                    response = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    pairs = self.extract_qa_pairs(response)
                    print(f"Fallback attempt extracted {len(pairs)} QA pairs")

                for q, a in pairs:
                    # Improved filtering logic
                    placeholder_phrases = [
                        "write a",
                        "specific detailed question",
                        "comprehensive answer",
                        "[question",
                        "[answer",
                        "question here",
                        "answer here",
                    ]

                    # Check if either question or answer has placeholder text
                    is_placeholder = False
                    for phrase in placeholder_phrases:
                        if phrase.lower() in q.lower() or phrase.lower() in a.lower():
                            is_placeholder = True
                            break

                    if (
                        not is_placeholder
                        and len(q.strip()) > 10
                        and len(a.strip()) > 20
                    ):  # Better length checks
                        qa_pairs.append(
                            {
                                "title": chunk["title"],
                                "chunk_id": chunk["chunk_id"],
                                "context": chunk["text"],
                                "question": q.strip(),
                                "answer": a.strip(),
                            }
                        )
                    else:
                        print(
                            f"Rejected pair - Q: {q[:30]}... ({len(q.strip())} chars), A: {a[:30]}... ({len(a.strip())} chars)"
                        )
                        if is_placeholder:
                            print("  Reason: Contains placeholder text")
                        else:
                            print("  Reason: Too short")
            except Exception as e:
                print(f"Error processing chunk {i + 1}: {e}")
                continue  # Skip this chunk but continue with others

        # Ensure we have at least some data
        if len(qa_pairs) == 0:
            # Try one more time with a different model if available, using a smaller chunk
            try:
                print("Trying with a different approach for at least some data...")
                # Take a small subset of chunks to ensure we get something
                small_chunks = chunks[: min(5, len(chunks))]

                # Manually create at least one QA pair as a last resort
                for chunk in small_chunks:
                    # Extract a simple question from first sentence
                    sentences = chunk["text"].split(". ")
                    if len(sentences) > 1:
                        first_sentence = sentences[0].strip()
                        # Create a "what" question from first sentence
                        words = first_sentence.split()
                        if len(words) > 5:
                            question = (
                                f"What does the text say about {' '.join(words[1:4])}?"
                            )
                            answer = (
                                first_sentence + ". " + sentences[1]
                                if len(sentences) > 1
                                else first_sentence
                            )

                            qa_pairs.append(
                                {
                                    "title": chunk["title"],
                                    "chunk_id": chunk["chunk_id"],
                                    "context": chunk["text"],
                                    "question": question,
                                    "answer": answer,
                                }
                            )
            except Exception as e:
                print(f"Emergency data creation also failed: {e}")
                # If all else fails, raise the error
                raise ValueError(
                    "No QA pairs were generated. Check the model outputs and extraction logic."
                )

        return qa_pairs

    def extract_qa_pairs(self, text):
        """Extract question-answer pairs with robust pattern matching"""
        # Try multiple regex patterns for different possible formats
        patterns = [
            # Standard format: Q1: question\nA1: answer
            r"Q(\d+)[\s:]+(.*?)[\s\n]+A\1[\s:]+(.*?)(?=[\s\n]+Q\d+[\s:]|$)",
            # Alternative format: Question 1: question\nAnswer 1: answer
            r"Question\s*(\d+)[\s:]+(.*?)[\s\n]+Answer\s*\1[\s:]+(.*?)(?=[\s\n]+Question\s*\d+[\s:]|$)",
            # Simple format: Q: question\nA: answer
            r"Q:[\s]+(.*?)[\s\n]+A:[\s]+(.*?)(?=[\s\n]+Q:[\s]|$)",
        ]

        all_pairs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

            # Process matches based on capture group structure
            pairs = []
            for match in matches:
                if len(match) == 3:  # Numbered format with 3 capture groups
                    _, question, answer = match
                elif len(match) == 2:  # Simple format with 2 capture groups
                    question, answer = match
                else:
                    continue

                question = question.strip()
                answer = answer.strip()

                # Filter out template placeholders
                if ("[" in question and "]" in question) or (
                    "[" in answer and "]" in answer
                ):
                    continue

                if question and answer:  # Ensure both are non-empty
                    pairs.append((question, answer))

            if pairs:  # If we found pairs with this pattern, add them
                all_pairs.extend(pairs)
                print(f"Pattern matched {len(pairs)} valid pairs")

        # Add detailed debugging output
        print(f"Total extracted: {len(all_pairs)} valid pairs")
        if len(all_pairs) == 0:
            print("DEBUG - Model response excerpt:")
            print(text[:500])  # Print beginning of response
            print("...")
            print(text[-500:])  # Print end of response

        return all_pairs


def create_synthetic_data(
    documents_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/dataset/q3_dataset",
    output_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
):
    # Process documents
    processor = DocumentProcessor(documents_dir)
    chunks = processor.chunk_documents()

    # Generate QA pairs
    generator = QAGenerator()
    qa_pairs = generator.generate_qa_pairs(chunks)

    # Create dataset
    dataset = Dataset.from_list(qa_pairs)

    # Create train/validation/test splits
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_valid = splits["train"]
    test = splits["test"]

    # Further split train into train and validation
    splits = train_valid.train_test_split(
        test_size=0.25, seed=42
    )  # 0.25 * 0.8 = 0.2 of original data
    train = splits["train"]
    validation = splits["test"]

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train.to_json(os.path.join(output_dir, "train.json"))
    validation.to_json(os.path.join(output_dir, "validation.json"))
    test.to_json(os.path.join(output_dir, "test.json"))

    print(
        f"Dataset created with {len(train)} training, {len(validation)} validation, and {len(test)} test examples."
    )
    return train, validation, test


class QAFineTuner:
    def __init__(self, model_name, data_dir, output_dir):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.validation_dataset = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load and preprocess the datasets."""
        train_path = os.path.join(self.data_dir, "train.json")
        validation_path = os.path.join(self.data_dir, "validation.json")

        self.train_dataset = load_dataset("json", data_files=train_path)["train"]
        self.validation_dataset = load_dataset("json", data_files=validation_path)[
            "train"
        ]

        print(
            f"Loaded {len(self.train_dataset)} training examples and {len(self.validation_dataset)} validation examples."
        )

    def prepare_model(self):
        """Load and prepare the model with LoRA."""
        # Clear memory before model loading
        gc.collect()
        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes

            print(f"Using bitsandbytes version: {bitsandbytes.__version__}")

            # Configure quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config,
                use_cache=False,  # Important for training
            )

            # Apply LoRA adapter
            self.model = get_peft_model(self.model, LORA_CONFIG)

            print("Successfully loaded model with 4-bit quantization and LoRA adapters")

        except (ImportError, ModuleNotFoundError) as e:
            print(f"Warning: Could not use quantization: {e}")
            print("Falling back to CPU loading with offloading")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            # Apply LoRA adapter
            self.model = get_peft_model(self.model, LORA_CONFIG)

            # Move to GPU selectively if possible
            try:
                self.model.to_bettertransformer()
            except:
                print("Could not convert to BetterTransformer")

        # Print trainable parameters info
        self.model.print_trainable_parameters()

    def format_instruction(self, example):
        """Format the input as an instruction."""
        context = example["context"]
        question = example["question"]
        answer = example["answer"]

        instruction = f"""### System:
You are an AI assistant that specializes in answering questions about AI research papers.
Your responses should be comprehensive, accurate, and based on the provided context.

### Human:
I have a question about an AI research paper.

Context: {context}

Question: {question}

### Assistant:
{answer}
"""
        return instruction

    def tokenize_function(self, examples):
        """Tokenize and format the examples."""
        instructions = []

        for i in range(len(examples["context"])):
            example = {
                "context": examples["context"][i],
                "question": examples["question"][i],
                "answer": examples["answer"][i],
            }
            instructions.append(self.format_instruction(example))

        tokenized = self.tokenizer(
            instructions,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def prepare_datasets(self):
        """Prepare tokenized datasets for training."""
        tokenize_batch_size = 8

        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=self.train_dataset.column_names,
        )

        self.validation_dataset = self.validation_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=self.validation_dataset.column_names,
        )

        print(f"Tokenized datasets: {self.train_dataset}, {self.validation_dataset}")

    def train(self):
        """Train the model."""
        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        # Initialize wandb for tracking
        wandb.init(project="qwen-ai-research-qa", name="qwen-2.5-3b-qlora")

        # Make sure no DeepSpeed configurations are active
        for key in list(os.environ.keys()):
            if "DEEPSPEED" in key or "DS_" in key:
                del os.environ[key]

        # Configure training arguments with NO DeepSpeed
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=50,
            eval_steps=1000,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="wandb",
            # Switch to standard FP32 precision
            bf16=False,
            fp16=False,
            # DeepSpeed settings - force disable
            deepspeed=None,
            local_rank=-1,
            ddp_backend=None,  # Don't use any distributed backend
        )

        # Create trainer with standard optimizer
        from transformers import AdamW

        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer with explicit optimizer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, None),  # Use our optimizer, no scheduler
        )

        # Train the model
        trainer.train()

        # Save the final model
        self.model.save_pretrained(os.path.join(self.output_dir, "final"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))

        print("Training complete!")


class ModelQuantizer:
    def __init__(
        self,
        model_path="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/fine_tuned_model/final",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        output_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model",
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.output_dir = output_dir
        self.quantized_model_path = os.path.join(output_dir, "model.gguf")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_and_merge_model(self):
        """Load the LoRA model and merge with the base model."""
        print("Loading base model...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load LoRA weights
        print("Loading and merging LoRA weights...")
        model = PeftModel.from_pretrained(base_model, self.model_path)

        # Merge LoRA weights with base model
        model = model.merge_and_unload()

        # Save merged model and tokenizer
        merged_model_path = os.path.join(self.output_dir, "merged")
        os.makedirs(merged_model_path, exist_ok=True)

        print(f"Saving merged model to {merged_model_path}...")
        model.save_pretrained(merged_model_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.save_pretrained(merged_model_path)

        print("Model and tokenizer saved successfully.")
        return merged_model_path

    def convert_to_gguf(self, merged_model_path):
        """Convert the merged model to GGUF format with 4-bit quantization."""
        print("Converting to GGUF format with 4-bit quantization...")

        # Check for existing GGUF model
        if os.path.exists(self.quantized_model_path):
            print(f"GGUF model already exists at {self.quantized_model_path}")
            user_input = input("Do you want to rebuild it? (y/n): ").lower()
            if user_input != "y":
                print("Using existing GGUF model.")
                return self.quantized_model_path

        # Clone llama.cpp repository if needed
        if not os.path.exists("llama.cpp"):
            try:
                print("Cloning llama.cpp repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error cloning llama.cpp repository.")
                raise RuntimeError("Failed to clone llama.cpp repository")

        # Build llama.cpp with better error handling
        try:
            print("Building llama.cpp with CMake (this may take a few minutes)...")
            os.makedirs("llama.cpp/build", exist_ok=True)

            # Configure with CMake
            subprocess.run(
                ["cmake", "-S", "llama.cpp", "-B", "llama.cpp/build"], check=True
            )

            # Build with CMake
            subprocess.run(
                ["cmake", "--build", "llama.cpp/build", "--parallel"], check=True
            )

            print("llama.cpp built successfully with CMake")

            # Use convert_hf_to_gguf.py with verbose output to see what's happening
            convert_script = "llama.cpp/convert_hf_to_gguf.py"

            if not os.path.exists(convert_script):
                print(f"ERROR: {convert_script} not found!")
                print("Please verify your llama.cpp installation.")
                raise RuntimeError(f"Conversion script not found: {convert_script}")

            print(f"\nRunning conversion script with enhanced debugging...")

            # Try conversion with detailed error output
            try:
                result = subprocess.run(
                    [
                        "python3",
                        convert_script,
                        merged_model_path,
                        "--outfile",
                        self.quantized_model_path,
                        "--outtype",
                        "q4_0",
                        "--verbose",  # Add verbose output
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)

            except subprocess.CalledProcessError as e:
                print("\n===== Conversion Error Details =====")
                print(f"Exit code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                print("===================================\n")

                print(
                    "Trying alternate conversion approach with arch-specific parameters..."
                )
                try:
                    # Try with explicit model architecture parameters
                    result = subprocess.run(
                        [
                            "python3",
                            convert_script,
                            merged_model_path,
                            "--outfile",
                            self.quantized_model_path,
                            "--outtype",
                            "q4_0",
                            "--model-type",
                            "llama",  # Try forcing llama architecture
                            "--ctx",
                            "4096",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(result.stdout)

                except subprocess.CalledProcessError as e2:
                    print(f"Alternate approach also failed")
                    print(f"STDOUT: {e2.stdout}")
                    print(f"STDERR: {e2.stderr}")
                    raise RuntimeError("All conversion methods failed")

        except Exception as e:
            print(f"Error during build or conversion process: {e}")
            raise RuntimeError("Failed to convert model to GGUF format")

        print(
            f"Model successfully converted to GGUF format: {self.quantized_model_path}"
        )

        # Copy tokenizer files to output directory
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
        for file in tokenizer_files:
            src_path = os.path.join(merged_model_path, file)
            if os.path.exists(src_path):
                dst_path = os.path.join(self.output_dir, file)
                shutil.copy2(src_path, dst_path)

        return self.quantized_model_path

    def quantize(self):
        """Perform the complete quantization process."""
        merged_model_path = self.load_and_merge_model()
        gguf_path = self.convert_to_gguf(merged_model_path)
        return gguf_path


class Evaluator:
    def __init__(
        self,
        model_path="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model/model.gguf",
        data_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
    ):
        self.data_dir = data_dir
        self.inference = ModelInference(model_path=model_path, use_rag=True)
        self.inference_no_rag = ModelInference(model_path=model_path, use_rag=False)

        # Download necessary NLTK data
        try:
            nltk.data.find("punkt")
        except LookupError:
            nltk.download("punkt")

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smooth = SmoothingFunction().method1

    def load_test_data(self):
        """Load the test dataset."""
        test_path = os.path.join(self.data_dir, "test.json")
        return load_dataset("json", data_files=test_path)["train"]

    def calculate_metrics(self, reference, candidate):
        """Calculate BLEU and ROUGE scores."""
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, candidate)

        # BLEU score
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        bleu_score = sentence_bleu(
            [reference_tokens], candidate_tokens, smoothing_function=self.smooth
        )

        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
        }

    def evaluate(self, sample_size=None):
        """Evaluate the model on the test set and save results to a JSON file."""
        test_data = self.load_test_data()

        # Limit evaluation to sample_size if specified
        if sample_size is not None:
            test_data = test_data.select(range(min(sample_size, len(test_data))))

        results_with_rag = []
        results_without_rag = []

        print(f"Evaluating on {len(test_data)} test examples...")

        for i, example in enumerate(test_data):
            print(f"Processing example {i + 1}/{len(test_data)}...")

            question = example["question"]
            reference_answer = example["answer"]

            # Generate answers with and without RAG
            answer_with_rag = self.inference.generate_answer(question)
            answer_without_rag = self.inference_no_rag.generate_answer(question)

            # Calculate metrics
            metrics_with_rag = self.calculate_metrics(reference_answer, answer_with_rag)
            metrics_without_rag = self.calculate_metrics(
                reference_answer, answer_without_rag
            )

            # Store results
            results_with_rag.append(
                {
                    "question": question,
                    "reference": reference_answer,
                    "prediction": answer_with_rag,
                    **metrics_with_rag,
                }
            )

            results_without_rag.append(
                {
                    "question": question,
                    "reference": reference_answer,
                    "prediction": answer_without_rag,
                    **metrics_without_rag,
                }
            )

        # Calculate average metrics
        avg_metrics_with_rag = {
            "bleu": sum(r["bleu"] for r in results_with_rag) / len(results_with_rag),
            "rouge1": sum(r["rouge1"] for r in results_with_rag)
            / len(results_with_rag),
            "rouge2": sum(r["rouge2"] for r in results_with_rag)
            / len(results_with_rag),
            "rougeL": sum(r["rougeL"] for r in results_with_rag)
            / len(results_with_rag),
        }

        avg_metrics_without_rag = {
            "bleu": sum(r["bleu"] for r in results_without_rag)
            / len(results_without_rag),
            "rouge1": sum(r["rouge1"] for r in results_without_rag)
            / len(results_without_rag),
            "rouge2": sum(r["rouge2"] for r in results_without_rag)
            / len(results_without_rag),
            "rougeL": sum(r["rougeL"] for r in results_without_rag)
            / len(results_without_rag),
        }

        print("\nEvaluation Results:")
        print("\nWith RAG:")
        for metric, value in avg_metrics_with_rag.items():
            print(f"{metric}: {value:.4f}")

        print("\nWithout RAG:")
        for metric, value in avg_metrics_without_rag.items():
            print(f"{metric}: {value:.4f}")

        # Prepare the overall results dictionary
        results = {
            "with_rag": {
                "detailed_results": results_with_rag,
                "average_metrics": avg_metrics_with_rag,
            },
            "without_rag": {
                "detailed_results": results_without_rag,
                "average_metrics": avg_metrics_without_rag,
            },
        }

        # Save the results to a JSON file named "metrics.json"
        with open(
            "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model/metrics.json",
            "w",
        ) as json_file:
            json.dump(results, json_file, indent=4)

        print("\nResults saved to metrics.json")
        return results


class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        # Use CPU for embeddings to save GPU memory
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # Get embedding dimension from the model
        self.embedding_dim = self.model.config.hidden_size

    def get_embedding_dim(self):
        """Return the embedding dimension of the model."""
        return self.embedding_dim

    def get_embeddings(self, texts: List[str], batch_size=16) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class VectorStore:
    def __init__(self, embedding_dim=768):
        self.index = faiss.IndexFlatL2(
            embedding_dim
        )  # L2 distance for similarity search
        self.texts = []

    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        """Add texts and their embeddings to the vector store."""
        # Add embeddings to index
        self.index.add(embeddings)
        # Store original texts
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for most similar texts given a query embedding."""
        # Reshape query embedding
        query_embedding = query_embedding.reshape(1, -1)

        # Search in the index
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts) and idx >= 0:
                results.append(
                    {
                        "text": self.texts[idx],
                        "score": float(distances[0][i]),
                        "id": int(idx),
                    }
                )

        return results


class RAGSystem:
    def __init__(
        self,
        data_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
    ):
        self.embedding_model = EmbeddingModel()
        # Use the actual embedding dimension from the model
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_model.get_embedding_dim()
        )
        self.data_dir = data_dir

    def build_index(self, force_rebuild=False):
        """Build the vector index from the dataset chunks."""
        index_file = os.path.join(self.data_dir, "vector_index.faiss")
        texts_file = os.path.join(self.data_dir, "vector_texts.npy")

        # Load from disk if exists and not forced to rebuild
        if (
            os.path.exists(index_file)
            and os.path.exists(texts_file)
            and not force_rebuild
        ):
            self.vector_store.index = faiss.read_index(index_file)
            self.vector_store.texts = np.load(texts_file, allow_pickle=True).tolist()
            print(
                f"Loaded existing index with {len(self.vector_store.texts)} documents."
            )
            return

        # Load datasets
        print("Building vector index...")

        # Load train, validation, test datasets
        train_path = os.path.join(self.data_dir, "train.json")
        validation_path = os.path.join(self.data_dir, "validation.json")
        test_path = os.path.join(self.data_dir, "test.json")

        train_data = load_dataset("json", data_files=train_path)["train"]
        validation_data = load_dataset("json", data_files=validation_path)["train"]
        test_data = load_dataset("json", data_files=test_path)["train"]

        # Combine all contexts
        all_contexts = []
        seen_contexts = set()

        # Helper to add unique contexts
        def add_unique_contexts(dataset):
            for item in dataset:
                context = item["context"]
                if context not in seen_contexts:
                    all_contexts.append(context)
                    seen_contexts.add(context)

        add_unique_contexts(train_data)
        add_unique_contexts(validation_data)
        add_unique_contexts(test_data)

        print(f"Found {len(all_contexts)} unique contexts.")

        # Generate embeddings
        embeddings = self.embedding_model.get_embeddings(all_contexts)

        # Add to vector store
        self.vector_store.add_texts(all_contexts, embeddings)

        # Save to disk
        faiss.write_index(self.vector_store.index, index_file)
        np.save(texts_file, np.array(self.vector_store.texts, dtype=object))

        print(f"Built and saved index with {len(all_contexts)} documents.")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant contexts for a query."""
        # Generate query embedding
        query_embedding = self.embedding_model.get_embeddings([query])[0]

        # Search in vector store
        results = self.vector_store.search(query_embedding, k=k)

        # Return contexts
        return [item["text"] for item in results]


class ModelInference:
    def __init__(
        self,
        model_path: str = "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model/model.gguf",
        use_rag: bool = True,
        context_length: int = 4096,
        num_retrieved_docs: int = 3,
    ):
        self.model_path = model_path
        self.use_rag = use_rag
        self.num_retrieved_docs = num_retrieved_docs

        # Initialize Llama model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_batch=512,
            n_gpu_layers=-1,  # Use all layers on GPU if available
        )

        # Initialize RAG system if needed
        if use_rag:
            self.rag = RAGSystem()
            self.rag.build_index()

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context using RAG with token count limiting."""
        if not self.use_rag:
            return ""

        contexts = self.rag.retrieve(query, k=self.num_retrieved_docs)

        # Calculate token budgets
        system_prompt = "You are an AI assistant that specializes in answering questions about AI research papers."
        query_prompt = f"Question: {query}"
        combined_prompt = system_prompt + query_prompt

        # Fix: Use the more reliable approach with llama_cpp
        # Reserve tokens for the system prompt, query, and generated response
        try:
            # Use the proper encoding with llama_cpp
            reserved_tokens = (
                len(self.llm.tokenize(bytes(combined_prompt, "utf-8"))) + 1024
            )
        except TypeError:
            # Fallback method if bytes conversion doesn't work
            reserved_tokens = len(combined_prompt.split()) * 2 + 1024  # Approximate

        max_context_tokens = self.llm.n_ctx() - reserved_tokens

        # Start with all contexts and trim as needed
        selected_contexts = []
        current_tokens = 0

        for context in contexts:
            try:
                context_tokens = len(self.llm.tokenize(bytes(context, "utf-8")))
            except TypeError:
                # Fallback approximation
                context_tokens = len(context.split()) * 2

            if current_tokens + context_tokens <= max_context_tokens:
                selected_contexts.append(context)
                current_tokens += context_tokens
            else:
                # Try to add a truncated version if it's the first context
                if len(selected_contexts) == 0:
                    # Estimate truncation point (rough approximation)
                    max_chars = int(max_context_tokens / context_tokens * len(context))
                    truncated = context[:max_chars]
                    selected_contexts.append(truncated)
                break

        return "\n\n".join(selected_contexts)

    def format_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Format the prompt for the model."""
        system_message = "You are an AI assistant that specializes in answering questions about AI research papers. Provide comprehensive, accurate responses based on the information available to you."

        if context:
            prompt = f"""### System:
{system_message}

### Human:
I have a question about an AI research paper.

Here is some relevant context:
{context}

Question: {query}

### Assistant:
"""
        else:
            prompt = f"""### System:
{system_message}

### Human:
Question about AI research: {query}

### Assistant:
"""
        return prompt

    def generate_answer(self, query: str) -> str:
        """Generate an answer for a query."""
        # Retrieve context if using RAG
        context = self.retrieve_context(query) if self.use_rag else None

        # Format prompt
        prompt = self.format_prompt(query, context)

        # Generate response
        start_time = time.time()
        response = self.llm(
            prompt,
            max_tokens=1024,
            stop=["### Human:", "### System:"],
            temperature=0.7,
            top_p=0.95,
        )
        end_time = time.time()

        # Extract answer text
        answer = response["choices"][0]["text"].strip()

        # Log performance
        print(f"Generation time: {end_time - start_time:.2f} seconds")

        return answer


class ModelQuantizer:
    def __init__(
        self,
        model_path="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/fine_tuned_model/final",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        output_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model",
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.output_dir = output_dir
        self.quantized_model_path = os.path.join(output_dir, "model.gguf")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_and_merge_model(self):
        """Load the LoRA model and merge with the base model."""
        print("Loading base model...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load LoRA weights
        print("Loading and merging LoRA weights...")
        model = PeftModel.from_pretrained(base_model, self.model_path)

        # Merge LoRA weights with base model
        model = model.merge_and_unload()

        # Save merged model and tokenizer
        merged_model_path = os.path.join(self.output_dir, "merged")
        os.makedirs(merged_model_path, exist_ok=True)

        print(f"Saving merged model to {merged_model_path}...")
        model.save_pretrained(merged_model_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.save_pretrained(merged_model_path)

        print("Model and tokenizer saved successfully.")
        return merged_model_path

    def convert_to_gguf(self, merged_model_path):
        """Convert the merged model to GGUF format with quantization."""
        print("Converting to GGUF format with quantization...")

        # Check for existing GGUF model
        if os.path.exists(self.quantized_model_path):
            print(f"GGUF model already exists at {self.quantized_model_path}")
            user_input = input("Do you want to rebuild it? (y/n): ").lower()
            if user_input != "y":
                print("Using existing GGUF model.")
                return self.quantized_model_path

        # Clone llama.cpp repository if needed
        if not os.path.exists("llama.cpp"):
            try:
                print("Cloning llama.cpp repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error cloning llama.cpp repository.")
                raise RuntimeError("Failed to clone llama.cpp repository")

        # Build llama.cpp with better error handling
        try:
            print("Building llama.cpp with CMake (this may take a few minutes)...")
            os.makedirs("llama.cpp/build", exist_ok=True)

            # Configure with CMake
            subprocess.run(
                ["cmake", "-S", "llama.cpp", "-B", "llama.cpp/build"], check=True
            )

            # Build with CMake
            subprocess.run(
                ["cmake", "--build", "llama.cpp/build", "--parallel"], check=True
            )

            print("llama.cpp built successfully with CMake")

            # Use convert_hf_to_gguf.py with verbose output to see what's happening
            convert_script = "llama.cpp/convert_hf_to_gguf.py"

            if not os.path.exists(convert_script):
                print(f"ERROR: {convert_script} not found!")
                print("Please verify your llama.cpp installation.")
                raise RuntimeError(f"Conversion script not found: {convert_script}")

            print(f"\nRunning conversion script with enhanced debugging...")

            # Try conversion with detailed error output - using q8_0 instead of q4_0
            try:
                result = subprocess.run(
                    [
                        "python3",
                        convert_script,
                        merged_model_path,
                        "--outfile",
                        self.quantized_model_path,
                        "--outtype",
                        "q8_0",  # Changed from q4_0 to q8_0
                        "--verbose",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)

            except subprocess.CalledProcessError as e:
                print("\n===== Conversion Error Details =====")
                print(f"Exit code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                print("===================================\n")

                print(
                    "Trying alternate conversion approach with arch-specific parameters..."
                )
                try:
                    # Try with explicit model architecture parameters - using q8_0
                    result = subprocess.run(
                        [
                            "python3",
                            convert_script,
                            merged_model_path,
                            "--outfile",
                            self.quantized_model_path,
                            "--outtype",
                            "q8_0",  # Changed from q4_0 to q8_0
                            "--model-name",
                            "Qwen",  # Added model name hint
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(result.stdout)

                except subprocess.CalledProcessError as e2:
                    print(f"Alternate approach also failed")
                    print(f"STDOUT: {e2.stdout}")
                    print(f"STDERR: {e2.stderr}")
                    raise RuntimeError("All conversion methods failed")

        except Exception as e:
            print(f"Error during build or conversion process: {e}")
            raise RuntimeError("Failed to convert model to GGUF format")

        print(
            f"Model successfully converted to GGUF format: {self.quantized_model_path}"
        )

        # Copy tokenizer files to output directory
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
        for file in tokenizer_files:
            src_path = os.path.join(merged_model_path, file)
            if os.path.exists(src_path):
                dst_path = os.path.join(self.output_dir, file)
                shutil.copy2(src_path, dst_path)

        return self.quantized_model_path

    def quantize(self):
        """Perform the complete quantization process."""
        merged_model_path = self.load_and_merge_model()
        gguf_path = self.convert_to_gguf(merged_model_path)
        return gguf_path


from google.colab import drive

drive.mount("/content/drive")

import os


def main():
    # Generate synthetic dataset with default directories
    print("Generating synthetic dataset...")
    create_synthetic_data(
        documents_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/dataset/q3_dataset",
        output_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
    )

    # Fine-tune the model using default parameters
    print("Fine-tuning Qwen/Qwen2.5-3B-Instruct...")
    fine_tuner = QAFineTuner(
        "Qwen/Qwen2.5-3B-Instruct",
        "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
        "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/fine_tuned_model",
    )
    fine_tuner.load_data()
    fine_tuner.prepare_model()
    fine_tuner.prepare_datasets()
    fine_tuner.train()

    # Quantize the model to GGUF format with default settings
    print("Quantizing the model...")
    quantizer = ModelQuantizer(
        model_path="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/fine_tuned_model/final",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        output_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model",
    )
    quantizer.quantize()

    # Build the RAG index with default directory
    print("Building RAG index...")
    rag = RAGSystem(
        data_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data"
    )
    rag.build_index(force_rebuild=True)

    # Evaluate the model using default settings (using all available samples)
    print("Evaluating the model...")
    evaluator = Evaluator(
        model_path=os.path.join(
            "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model",
            "model.gguf",
        ),
        data_dir="/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/data",
    )
    evaluator.evaluate(sample_size=None)

    # Run inference using default parameters and a sample query
    print("Running inference...")
    inference = ModelInference(
        model_path=os.path.join(
            "/content/drive/MyDrive/LLM-Fine-tuning-Challenge-Enhancing-Qwen-2.5-3B-for-AI-Research-QA/quantized_model",
            "model.gguf",
        ),
        use_rag=True,
    )
    default_query = "What is the latest research in AI?"
    answer = inference.generate_answer(default_query)
    print(f"\nQuery: {default_query}\n")
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
