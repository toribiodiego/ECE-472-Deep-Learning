

> This repository contains the code for **ECE 472: Deep Learning**, a 3-credit graduate course at The Cooper Union for the Advancement of Science and Art, providing comprehensive exposure to modern deep learning techniques and their applications.

## Deep Learning
**Course, Fall 2023**  
**Instructor:** Professor Chris Curro


### Overview

This course introduces students to deep learning concepts and methodologies, emphasizing practical understanding through computational exercises and project-based learning. The curriculum spans foundational topics such as regression, gradient descent, and multi-layer perceptrons, as well as advanced models including convolutional neural networks, transformers, neural ODEs, and diffusion models. Through a series of programming assignments, quizzes, and class discussions on seminal research papers, students develop both theoretical knowledge and hands-on experience necessary for conducting innovative deep learning research and real-world problem-solving.



### Material

The primary reference for this course is [*Deep Learning*](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, providing essential foundations in:

- Neural networks and gradient-based learning
- Convolutional neural networks and related architectures
- Attention mechanisms and transformer models
- Advanced techniques such as diffusion models, neural ODEs, and fine-tuning strategies

Additionally, relevant research papers will be provided throughout the semester to deepen understanding and facilitate project work.


#### Repository Structure

```markdown
.
├── Assignment 1
│   ├── basis_expansion.py
│   └── linear.py
├── Assignment 2
│   ├── MLP.py
│   ├── basis_expansion.py
│   └── linear.py
├── Assignment 3
│   └── cnn.py
├── Assignment 4
│   ├── CIFAR10.py
│   └── CIFAR100.py
├── Assignment 5
│   └── hw5.py
├── Assignment 6
│   └── hw6.py
├── Assignment 7
│   └── sine.py
└── README.md
```

- **Assignment 1** – *Gaussian‑Basis Linear Regression*  
  - Fits a noisy sine wave by projecting inputs through radial‑basis functions (`basis_expansion.py`) and training a lightweight dense layer (`linear.py`). Emphasises feature engineering, closed‑form loss visualisation, and learning‑rate scheduling.

- **Assignment 2** – *Spiral MLP Classifier*   
  - Explores nonlinear decision boundaries using a configurable MLP (`MLP.py`). Shared utilities provide Gaussian feature mapping and baseline linear comparisons, while the training loop demonstrates weight initialisation, dropout, and metric logging.

- **Assignment 3** – *MNIST*  
  - Implements a small CNN (`cnn.py`) with two convolutional blocks, batch normalisation, and max‑pooling. Includes data loaders, simple augmentation, and accuracy tracking to highlight fundamentals of image classification.

- **Assignment 4** – *CIFAR*
  - Parallel pipelines (`CIFAR10.py`, `CIFAR100.py`) benchmark a residual architecture on both CIFAR datasets. Each script handles advanced augmentation, learning‑rate warm‑up, and early stopping to illustrate scalable image‑classification workflows.

- **Assignment 5** – *Text Classification with AG News*  
  - Encodes news headlines with Sentence‑BERT and trains a logistic‑regression head (`hw5.py`). Covers text cleaning, dataset stratification, learning‑rate reduction on plateau, and macro‑averaged evaluation metrics.

- **Assignment 6** – *Transformer*  
  - `hw6.py` assembles positional encoding, multi‑head attention, feed‑forward layers, and a custom Adam optimiser into a mini‑transformer. Trains on a synthetic token dataset to demonstrate sequence modelling, mask handling, and autoregressive inference.

- **Assignment 7** – *SIREN Image Reconstruction*  
  - Trains a sinusoidal‑representation network (`sine.py`) that maps 2‑D coordinates to RGB values, recreating images at arbitrary resolution. Highlights implicit neural representations, spectral bias, and qualitative evaluation via side‑by‑side plots.




#### Reproducing Work

1. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```
2. **Install Dependencies:**
   Once inside an assignment’s directory (e.g., `Assignment 1`), run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Assignment Script:**
   Execute the main script for the assignment:
   ```bash
   python basis_expansion.py
   ```