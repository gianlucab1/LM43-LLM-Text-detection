# AI-Generated Text Detection

This repository contains the code developed for my Master's thesis in "Text Sciences for Digital Professions" titled: **"AI-Generated Text Detection: A Comparative Analysis of White-Box and Black-Box Methods"**.

## 🎯 The Project in Brief

As models like ChatGPT become more widespread, telling the difference between human-written and AI-generated text is a growing challenge. This project explores this field by analyzing and comparing the main strategies for detection, following the path of my thesis.

The goal was to analyze two fundamental approaches:
1.  **White-Box Methods**: Techniques that can "look inside" the AI model to analyze its internal characteristics (such as generation statistics).
2.  **Black-Box Methods**: A more realistic approach, where a specialized classifier is trained to recognize AI text based solely on the final output, without any access to the model that generated it.

## 🔬 Scripts Overview

Each script in this repository corresponds to one of the key experiments from the thesis.

*   `detection_perplexity.py`
    *   **What it does:** Implements a White-Box method that measures how "surprised" a model is by a text. The hypothesis is that AI-generated texts are less "surprising" (and thus have a lower perplexity) to the model itself.

*   `detection_log_probability.py`
    *   **What it does:** Another White-Box method, closely related to the previous one, which calculates the average probability of each word in a text according to the model.

*   `detection_watermark.py`
    *   **What it does:** Experiments with an "active" White-Box technique. It inserts a statistical "signature"—invisible to the human eye—during text generation and then attempts to detect it.

*   `detection_classificatore_fine_tuned.py`
    *   **What it does:** Represents the Black-Box approach. Here, a model (RoBERTa) is trained to become a "detective," learning to distinguish human from artificial texts. It is then tested on both in-domain data and completely new data (the M4 dataset) to evaluate its reliability.

## ⚙️ How to Use the Code

To run these experiments, follow these steps.

### 1. Set Up the Environment

First, clone the repository to your local machine:
```bash
git clone https://github.com/gianlucab1/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Download the Data

The scripts expect the data to be located in a specific folder.
*   Download the **OpenLLMText** dataset.
*   Create a folder named `OpenTextLLM` in the project's root directory and place the dataset folders (e.g., `Human`, `ChatGPT`, etc.) inside it.

### 3. Install Dependencies

It is recommended to use a virtual environment. Install the necessary libraries with this command:
```bash
pip install transformers torch datasets scikit-learn pandas scipy matplotlib```
*(Tip: You can create a `requirements.txt` file with this list for a cleaner installation).*

### 4. Run the Scripts

You can launch each experiment by running its corresponding Python script from your terminal. For example:
```bash
# To run the perplexity analysis
python detection_perplexity.py

# To train and evaluate the classifier
python detection_classificatore_fine_tuned.py
```

## Important Notes - Limitations & Context

*   **Academic Code**: The code in this repository was developed for academic research purposes for my thesis. It is not intended to be a production-ready software tool.
*   **Hardware Requirements**: These scripts require significant hardware, specifically a **GPU with a substantial amount of VRAM** (experiments were conducted on GPUs like the NVIDIA A100/T4). Running them on a standard CPU may be extremely slow or unfeasible.
*   **Manual Data Setup**: The datasets (e.g., OpenLLMText) **are not downloaded automatically**. You must download them manually and place them in the folder structure expected by the scripts (e.g., `./OpenTextLLM`).
*   **Scope of Research**: The results presented are specific to the models and datasets used in this analysis. The field of AI text detection is rapidly evolving, and performance may vary significantly with different models and approaches.

