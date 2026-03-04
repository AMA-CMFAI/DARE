<img width="3200" height="1312" alt="Gemini_Generated_Image_h25dizh25dizh25d (3)" src="https://github.com/user-attachments/assets/2412173f-3045-4b16-9105-d5540eb530ea" />

<div align="center">

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/Stephen-SMJ/DARE-R-Retriever)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/Stephen-SMJ/RPKB)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

> This repository contains the official code, model, and database for the paper: **Enhancing LLM Agents with Distribution-Conditional Retrieval for Data Analysis**.

Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. 

We propose **DARE (Distribution-Aware Retrieval Embedding)**, a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval.

## 🌟 Key Contributions

1. **RPKB (R Package Knowledge Base):** A curated database derived from 8,191 high-quality CRAN packages, complete with statistical metadata (Data Profiles).
2. **DARE Model:** A specialized bi-encoder model fine-tuned to fuse distributional features with function metadata, improving retrieval relevance by up to 17% (NDCG@10) over state-of-the-art open-source embeddings.
3. **RCodingAgent:** An R-oriented LLM agent designed for reliable R code generation, validated on a comprehensive suite of downstream statistical analysis tasks.

---

## 🚀 Quick Start (Zero-Configuration Inference)

We have open-sourced both our embedding model and the pre-computed ChromaDB database on Hugging Face. You can run distribution-aware function retrieval out of the box.

### 1. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/AMA-CMFAI/DARE.git
cd DARE
pip install -r requirements.txt
```

### 2. Run the DARE Retriever

The following script automatically downloads the DARE model and the RPKB database from Hugging Face and performs a distribution-aware search.

```python
# retrieval.py
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import os

# 1. Load DARE Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Stephen-SMJ/DARE-R-Retriever", trust_remote_code=False)
model.to(device)

# 2. Download and Connect to RPKB Database
db_dir = "./rpkb_db"
if not os.path.exists(os.path.join(db_dir, "DARE_db")):
    print("Downloading RPKB Database from Hugging Face...")
    snapshot_download(repo_id="Stephen-SMJ/RPKB", repo_type="dataset", local_dir=db_dir, allow_patterns="DARE_db/*")

client = chromadb.PersistentClient(path=os.path.join(db_dir, "DARE_db"))
collection = client.get_collection(name="inference")

# 3. Perform Search
query = "I have a sparse matrix with high dimensionality. I need to perform PCA."
query_embedding = model.encode(query, convert_to_tensor=False).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas"]
)

# Display Results
for rank, (doc_id, meta) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
    print(f"[{rank + 1}] Package: {meta.get('package_name')} :: Function: {meta.get('function_name')}")

```


[//]: # (## 📂 Repository Structure)

[//]: # ()
[//]: # (* `agent/`: Source code for the RCodingAgent framework.)

[//]: # (* `retrieval/`: Scripts for DARE embedding generation, ChromaDB re-indexing, and filtering.)

[//]: # (* `evaluation/`: Benchmark scripts for computing Recall, NDCG, and MRR.)

[//]: # (* `requirements.txt`: Project dependencies.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📖 Citation)

[//]: # ()
[//]: # (If you find DARE, RPKB, or RCodingAgent useful in your research, please cite our work:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{sun2026dare,)

[//]: # (  title={Enhancing LLM Agents with Distribution-Conditional Retrieval for Data Analysis},)

[//]: # (  author={Sun, Stephen and others},)

[//]: # (  journal={arXiv preprint arXiv:XXXX.XXXXX},)

[//]: # (  year={2026})

[//]: # (})
