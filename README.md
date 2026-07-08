<div align="center">
  <img src="figure/logo.svg" alt="SCPInt" width="640">
</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)]()

</div>

---

## 📋 Abstract

Mass spectrometry-based **s**ingle-**c**ell **p**roteomics (SCP) enables direct and unbiased measurement of cellular functional states, but its inherently low throughput necessitates **int**egration across multiple datasets to support downstream analysis and atlas-scale studies. Existing single-cell integration methods were primarily developed for count-based sequencing data and do not adequately model SCP measurements, which exhibit continuous distributions, heterogeneous missing structures, and substantial platform- and preparation-specific technical variation.

**SCPInt** is a deep-learning framework that learns a **unified biological representation** across heterogeneous MS-based SCP datasets while **explicitly disentangling and quantifying technical variation**. SCPInt combines a two-component Gaussian mixture model tailored to SCP expression characteristics with an adversarial architecture that separately encodes biological information and batch-associated information into biological embeddings and batch embeddings.

> Across **6 integration tasks** comprising **18 datasets** from **8 independent studies** and spanning multiple technologies, developmental stages, and sample preparation conditions, SCPInt consistently outperformed existing approaches in preserving biological structure and establishing unified cross-dataset representations.

---

## 🔬 Key Contributions

<table>
<tr>
<td width="50%">

- 🧬 **Dedicated SCP modeling** — Two-component Gaussian mixture model specifically designed for SCP expression characteristics (continuous values, heterogeneous missingness)

- 🎯 **Explicit disentanglement** — Adversarial architecture that separates biological signals from batch-associated variation into distinct embedding spaces

- 📊 **Quantitative technical phenotypes** — Batch embeddings provide interpretable measurements of platform bias, cryopreservation effects, and cell-type-specific technical sensitivity

</td>
<td width="50%">

- 🧠 **Atlas-scale integration** — Preserved continuous developmental manifolds in human brain SCP data, enabling proteomic trajectory reconstruction

- 🔬 **Biological discovery** — Identified biologically meaningful macrophage activation states across datasets

- ✅ **Comprehensive benchmarking** — Systematically evaluated across 18 datasets, 8 studies, and multiple technology platforms

</td>
</tr>
</table>

---

## 📖 Table of Contents

- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorials](#tutorials)
- [Data Availability](#data-availability)
- [Repository Structure](#repository-structure)
- [Benchmark Tasks](#benchmark-tasks)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## 🏗️ Model Architecture

![SCPInt Model Structure](figure/figures1.png)

SCPInt employs an adversarial autoencoder architecture with three core components:

1. **Encoder** — Maps input proteomic profiles into a latent representation
2. **Biological Decoder** — Reconstructs the expression profile from biological embeddings, using a two-component GMM to model both detected signals and missing values
3. **Batch Classifier** — Adversarially trained to predict batch identity from biological embeddings, enforcing batch-invariant representations via gradient reversal

---

## 💻 Installation

### Prerequisites

| Package  | Version     |
|----------|-------------|
| Python   | ≥ 3.10      |
| PyTorch  | 2.6.0+cu118 |
| NumPy    | 2.2.6       |
| Scanpy   | 1.11.5      |
| Pandas   | 2.3.3       |

### Hardware

A single consumer GPU (tested on NVIDIA RTX 2080Ti, 11 GB VRAM) is sufficient to run all experiments.

### Setup

```bash
# Clone the repository
git clone https://github.com/YuzhiSun/SCPInt.git
cd SCPInt

# Create a conda environment
conda create -n scpint python=3.10
conda activate scpint

# Install PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy==2.2.6 scanpy==1.11.5 pandas==2.3.3
```

---

## 🚀 Quick Start

```python
from code.SCPIntModel import AnnDataProcessor, scProteoIntegrator, Trainer
import anndata as ad

# 1. Load your data
adata = ad.read_h5ad("your_data.h5ad")

# 2. Preprocess & create data loader
processor = AnnDataProcessor(adata, batch_key="batch")
dataloader = processor.make_dataloader(batch_size=128)

# 3. Initialize model and trainer
model = scProteoIntegrator(
    n_genes=processor.n_genes,
    n_batches=processor.n_batches
)
trainer = Trainer(model, lr=1e-3)

# 4. Train
trainer.fit(dataloader, n_epochs=200)

# 5. Extract biological embeddings
X_tensor, _ = processor.to_tensors()
bio_emb, batch_emb = trainer.encode(X_tensor)
adata.obsm["X_scpint"] = bio_emb.numpy()
```

---

## 📚 Tutorials

We provide step-by-step Jupyter notebooks for each integration task:

| Task | Train | Analysis |
|------|-------|----------|
| Cross-technology integration | [`Train_task_cross_tech.ipynb`](code/Train_task_cross_tech.ipynb) | [`Product_cross_tech.ipynb`](code/Product_cross_tech.ipynb) |
| Multi-batch integration | [`Train_task_multibach.ipynb`](code/Train_task_multibach.ipynb) | [`Product_multibatch.ipynb`](code/Product_multibatch.ipynb) |
| Three-technology integration | [`Train_task_three_tech.ipynb`](code/Train_task_three_tech.ipynb) | [`Product_three_tech.ipynb`](code/Product_three_tech.ipynb) |
| Human brain SCP atlas | [`Train_task_human_brain_scp.ipynb`](code/Train_task_human_brain_scp.ipynb) | [`Product_human_brain_scp.ipynb`](code/Product_human_brain_scp.ipynb) |
| Macrophage LPS activation | [`Train_task_lps.ipynb`](code/Train_task_lps.ipynb) | [`Product_lps.ipynb`](code/Product_lps.ipynb) |
| Frozen vs. fresh comparison | [`Train_task_frozen_fresh.ipynb`](code/Train_task_frozen_fresh.ipynb) | [`Product_frozen_fresh.ipynb`](code/Product_frozen_fresh.ipynb) |

Each notebook is self-contained with detailed comments and expected outputs.

---

## 📦 Data Availability

All datasets used in this study are publicly available on Zenodo:

> [![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20537625-blue?logo=zenodo)](https://doi.org/10.5281/zenodo.20537625)

After downloading, place the data files under the `data/` directory. See the [data README](data/README.md) for detailed instructions.

---

## 📁 Repository Structure

```
SCPInt/
├── code/
│   ├── SCPIntModel.py                 # Core model implementation
│   ├── Train_task_cross_tech.ipynb    # Training: cross-technology
│   ├── Product_cross_tech.ipynb       # Analysis: cross-technology
│   ├── Train_task_multibach.ipynb     # Training: multi-batch
│   ├── Product_multibatch.ipynb       # Analysis: multi-batch
│   ├── Train_task_three_tech.ipynb    # Training: three technologies
│   ├── Product_three_tech.ipynb       # Analysis: three technologies
│   ├── Train_task_human_brain_scp.ipynb  # Training: brain atlas
│   ├── Product_human_brain_scp.ipynb     # Analysis: brain atlas
│   ├── Train_task_lps.ipynb           # Training: LPS activation
│   ├── Product_lps.ipynb              # Analysis: LPS activation
│   ├── Train_task_frozen_fresh.ipynb  # Training: frozen vs. fresh
│   └── Product_frozen_fresh.ipynb     # Analysis: frozen vs. fresh
├── data/
│   └── README.md                      # Data download instructions
├── figure/
│   ├── logo.svg                       # Project logo
│   └── figures1.png                   # Model architecture diagram
├── README.md
└── LICENSE
```

---

## 📊 Benchmark Tasks

SCPInt was evaluated across six diverse integration scenarios:

| # | Task | Datasets | Biological Context |
|---|------|----------|-------------------|
| 1 | Cross-technology | 4 datasets | Same cell type across different MS platforms |
| 2 | Multi-batch | 4 datasets | Multiple batches from the same technology |
| 3 | Three-technology | 3 datasets | Integration across three distinct MS technologies |
| 4 | Human brain atlas | 4 datasets | Developmental trajectory across brain regions |
| 5 | LPS activation | 3 datasets | Macrophage activation states |
| 6 | Frozen vs. fresh | 2 datasets | Cryopreservation effect on proteomic profiles |

---

## 📝 Citation

If you use SCPInt in your research, please cite our work:

```bibtex
@article{sun2025scpint,
  title   = {SCPInt: explicit disentanglement of biological and batch variation
             for single-cell proteomics integration},
  author  = {Sun, Yuzhi and ...},
  journal = {Under review},
  year    = {2025}
}
```

*(The complete citation will be updated upon publication.)*

---

## 📬 Contact

For questions, suggestions, or collaboration inquiries, please reach out:

- **Yuzhi Sun** — [yuzhi@stu.hit.edu.cn](mailto:yuzhi@stu.hit.edu.cn)
- **GitHub Issues** — [Open an issue](https://github.com/YuzhiSun/SCPInt/issues)

We welcome contributions and feedback from the community.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

  > **If you find SCPInt useful, please ⭐ star this repository!**

</div>
