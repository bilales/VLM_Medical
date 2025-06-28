# Fine-Tuning Qwen2.5 for Medical VQA: An AI Scientist's Approach

## 1. Project Overview

This project provides a complete, research-grade pipeline for fine-tuning the `Qwen/Qwen2.5-0.5B-Chat` model for Medical Visual Question Answering (VQA) on the `giskai/medvqa-2019` dataset. It is designed to be a comprehensive portfolio piece for an AI Scientist position, demonstrating not just model training but also systematic experimentation, deep analysis, and MLOps best practices.

The entire workflow is optimized to run on a single consumer GPU with 8GB of VRAM using QLoRA.

## 2. Core Features

- **Systematic Experimentation**: Integrated **Hyperparameter Optimization (HPO)** using Optuna to scientifically determine the best model parameters.
- **Deep Model Analysis**: Includes scripts for both quantitative evaluation (**BLEU, ROUGE**) and qualitative **Error Analysis** and **Explainability (XAI)** via attention visualization.
- **Efficient Fine-Tuning**: Leverages **QLoRA** and 4-bit quantization to make training feasible on limited hardware.
- **Experiment Tracking**: Uses **Weights & Biases (`wandb`)** to log all training metrics, configurations, and HPO results.
- **CI/CD & MLOps**: A **GitHub Actions** workflow ensures code quality through automated linting and testing.
- **Modular & Configurable**: The code is highly modular and centrally configured via `config.py`.

## 3. Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd qwen-medical-vqa
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Login to Weights & Biases:**
    ```bash
    wandb login
    ```

## 4. Usage

### 4.1. Hyperparameter Optimization

Before full training, run an HPO study to find the best parameters. This uses a small subset of the data for speed.

```bash
python run_hpo.py