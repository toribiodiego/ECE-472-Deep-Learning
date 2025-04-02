

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



### Repository Structure

Below is a revised section that includes the repository tree and a concise explanation on how to reproduce the work. This version assumes you’re working inside each assignment's directory:

---

### Repository Structure

```
.
├── Assignment 1
│   ├── artifacts
│   ├── basis_expansion.py
│   ├── config_noisy_sine.yaml
│   ├── linear.py
│   ├── requirements.txt
│   └── tests
├── Assignment 2
│   ├── MLP.py
│   ├── artifacts
│   ├── basis_expansion.py
│   ├── config.yaml
│   ├── linear.py
│   ├── requirements.txt
│   └── tests
├── Assignment 3
│   ├── cnn.py
│   ├── config.yaml
│   ├── requirements.txt
│   └── tests
├── Assignment 4
│   ├── 10.yaml
│   ├── 100.yaml
│   ├── CIFAR10.py
│   ├── CIFAR100.py
│   ├── requirements.txt
│   ├── response.pdf
│   └── test.py
├── Assignment 5
│   ├── config.yaml
│   ├── hw5.py
│   ├── requirements.txt
│   └── results.pdf
├── Assignment 6
│   ├── hw6.py
│   ├── output.txt
│   └── test.py
├── Assignment 7
│   ├── requirements.txt
│   └── sine.py
└── README.md
```

Each assignment folder is self-contained with its own `requirements.txt` file and a configuration file (`config.yaml`) for customizing hyperparameters and assignment-specific settings.

#### How to Reproduce the Environment

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

### Assignments

- **Assignment 1:** Linear Regression with Gaussian Basis Functions
- **Assignment 2:** Multi-layer Perceptron for Spiral Classification
- **Assignment 3:** CNN Classification of MNIST Digits
- **Assignment 4:** CNN Classification on CIFAR10 and CIFAR100
- **Assignment 5:** Text Classification using AG News Dataset
- **Assignment 6:** Implementation of Multi-Head Attention and Transformer Blocks
- **Assignment 7:** Image Fitting and Exploration with SIREN
