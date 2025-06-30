# Aerosol_CDT_summer_training

# 📘 Medical Imaging Classification & Retrieval-Augmented Generation & M

This repository contains hands-on notebooks focused on two complementary areas of applied machine learning:

## 🧠 1. Medical Imaging with Deep Learning & Transfer Learning

Work through the complete pipeline of classifying chest X-ray images using convolutional neural networks and transfer learning. This includes:

- Data loading and preprocessing from the COVID‑19 Radiography dataset
- Model training using Keras (CNNs & EfficientNet/MobileNet)
- Evaluation with Grad-CAM and confusion matrices
- Hyperparameter tuning with Keras Tuner

> 📌 **Notebook**: `Radiography_classifier-2.ipynb`
>
> 🖼️ **Dataset Access Notice**  
The radiography dataset used in this project will be **provided privately during the CDT programme** and must **not be redistributed or shared** publicly.
> If you wish to access it independently, the dataset is publicly available at:  
👉 https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

---

## 🔍 2. Retrieval-Augmented Generation (RAG) with Language Models

Explore how to build an intelligent RAG pipeline that can retrieve relevant documents and generate accurate, grounded responses using state-of-the-art language models. The notebook walks through:

- Text embeddings and vector stores (e.g., FAISS)
- Prompt construction and document retrieval
- End-to-end integration with RAG-style pipelines

> 📌 **Notebook**: `RAG_example-2.ipynb`

> 📄 *Note: Academic PDF content used in this example will be shared during the CDT. Details and citations will be added here.*

---


## 🚀 Course Requirements

All sessions and notebooks in this course will be delivered using **Google Colab**.

> ✅ You must have a **Google account** to participate.

### 📁 Uploading Data

Before running the notebooks, please upload the required datasets to your own Google Drive.

📤 **Upload link**: [Google Drive Upload Folder](https://drive.google.com)  
_Use the provided zip or folder download shared by the CDT, then upload it to your Drive manually._


---

## 🔐 API Key Requirements

To run all features of these notebooks, you'll need access to external APIs:

### ✅ OpenAI API Key
- Required for language model inference (e.g., GPT-based retrieval/generation).
- **This key will be provided to you by the CDT.**
- 🔍 **What to look for**:
  ```python
  os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
  ```

### ✅ Hugging Face Access Token
- Required to access certain models or datasets through the Hugging Face Hub.
- You must **create your own** token at: https://huggingface.co/settings/tokens
- 🔍 **What to look for to insert the token in the notebook**:
  ```python
  os.environ["HF_TOKEN"] = "your-huggingface-token-here"
  ```

Once you have your keys:
- **Never commit API keys to public repositories. The CDT provided OpenAI key will be disabled shortly after the course.**
