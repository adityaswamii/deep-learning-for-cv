# Deep Learning for Computer Vision

A structured, hands-on study of deep learning theory and implementation, combining Stanford's [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/), Christopher M. Bishop's [*Deep Learning*](https://www.bishopbook.com/), and coursework under **Prof. Sandesh Kamath**.

The goal of this repository is not just to follow along — it is to implement concepts from scratch, understand the mathematics behind them, and build intuition for why the field looks the way it does.

---

## Sources

| Source | Focus |
|---|---|
| [Stanford CS231n](https://cs231n.stanford.edu/) | Computer vision pipeline, CNNs, practical deep learning |
| Bishop — *Deep Learning* (2024) | Mathematical foundations, probabilistic framing |
| Prof. Sandesh Kamath | Guided coursework, structured problem sets |

---

## Repository Structure

```
deep-learning-for-cv/
│
├── chapter_1/          # Image classification: kNN, linear classifiers, loss functions
├── chapter_2/          # Neural networks: forward/backward pass, optimization
│
├── lectures/           # Lecture-aligned notebooks and supporting code
│   └── data/           # Dataset loading scripts (data itself is gitignored)
│
├── figures/            # Saved plots and visualizations
│
└── .gitignore          # Excludes raw datasets (CIFAR-10, MNIST) and model weights (*.pt)
```

---

## Progress

### CS231n

| Topic | Status |
|---|---|
| Image Classification & the Data-Driven Approach | ✅ |
| k-Nearest Neighbour Classifier | ✅ |
| Linear Classification: SVM & Softmax | ✅ |
| Optimization & Gradient Descent | ✅ |
| Neural Networks: Architecture & Activations | ✅ |
| Backpropagation & Computational Graphs | ✅ |
| Training Neural Networks | 🔄 In progress |
| Convolutional Neural Networks | 🔄 In progress |

### Bishop — *Deep Learning*

| Chapter | Topic | Status |
|---|---|---|
| 1 | The Deep Learning Revolution | ✅ |
| 2 | Probabilities | ✅ |
| 3 | Standard Machine Learning | ✅ |
| 4 | Single-Layer Networks | 🔄 In progress |

---

## Key Implementations

### Chapter 1 — Image Classification
- **kNN classifier** built from scratch using only NumPy; benchmarked with L1 and L2 distance on CIFAR-10
- **SVM loss (Hinge loss)** implemented with vectorised gradient computation
- **Softmax loss (Cross-entropy)** implemented alongside SVM for comparison
- Hyperparameter search across learning rates and regularisation strengths using k-fold cross-validation

### Chapter 2 — Neural Networks & Optimization
- **Two-layer neural network** with forward and backward pass implemented manually — no autograd
- Gradient checking via numerical differentiation to verify analytic gradients
- **Optimisers**: vanilla SGD, SGD with momentum, RMSProp, Adam — implemented from scratch and compared on a toy problem
- **PyTorch training loop** for a feedforward network trained on MNIST (weights excluded via `.gitignore`)

---

## Running the Notebooks

```bash
# 1. Clone the repository
git clone https://github.com/adityaswamii/deep-learning-for-cv.git
cd deep-learning-for-cv

# 2. Install dependencies
pip install numpy matplotlib jupyter torch torchvision

# 3. Download datasets
# CIFAR-10 and MNIST are loaded automatically by torchvision or via the
# download scripts in lectures/data/ — they are excluded from version control

# 4. Launch Jupyter
jupyter notebook
```

> **Python version:** 3.10+ recommended. All notebooks use standard scientific Python (NumPy, Matplotlib) plus PyTorch where noted.

---

## Notes on Approach

The notebooks in this repo prioritise understanding over convenience. Where a library function exists (e.g. `torch.nn.Linear`), the approach here is to implement it manually first, verify it works, and *then* cross-reference the library implementation. The figures saved in `/figures` are outputs from these experiments.

This is an active, ongoing study — progress and implementations are updated as new material is covered.
