# Deep Learning for Computer Vision

A structured, hands-on study of deep learning theory and implementation, combining Stanford's [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/), Christopher M. Bishop's [*Deep Learning*](https://www.bishopbook.com/), and independent coursework under **Prof. Sandesh Kamath**.

The goal of this repository is not just to follow along — it is to implement concepts from scratch, understand the mathematics behind them, and build intuition for why the field looks the way it does.

---

## Sources

| Source | Focus |
|---|---|
| [Stanford CS231n](https://cs231n.stanford.edu/) | Computer vision pipeline, practical deep learning |
| [Bishop — *Deep Learning* (2024)](https://www.bishopbook.com/) | Mathematical foundations |
| Prof. Sandesh Kamath | Independent study supervisor |

---

## Study Method

Each week, material from the lectures and assignments was presented back to Prof. Sandesh Kamath on a whiteboard — without notes. This forced active recall and required translating mathematical formalism into spoken intuition: not just knowing that backpropagation applies the chain rule, but being able to draw the computation graph, explain why gradients flow the way they do, and connect it to what the network is actually learning. The notebooks in this repo are the implementation side of that process.

---

## Repository Structure

```
deep-learning-for-cv/
│
├── chapter_1/          # Bishop Ch. 1–2 aligned notebooks: probability, classification foundations
├── chapter_2/          # Neural network implementation: forward/backward pass, optimisers
│
├── lectures/           # CS231n lecture-aligned notebooks and supporting code
│   └── data/           # Dataset loading utilities (datasets excluded from version control)
│
├── figures/            # Saved plots and visualisations from experiments
│
└── .gitignore          # Excludes raw datasets (CIFAR-10, MNIST) and model weights (*.pt)
```

---

## Progress

### CS231n

| Topic | Status |
|---|---|
| Computer Vision Overview | ✅ |
| Image Classification & the Data-Driven Approach | ✅ |
| k-Nearest Neighbour Classifier | ✅ |
| Linear Classification: SVM & Softmax | ✅ |
| Optimization & Gradient Descent | ✅ |
| Neural Networks: Architecture & Activations | ✅ |
| Backpropagation & Computational Graphs | ✅ |
| Training Neural Networks | ✅ |
| Convolutional Neural Networks | ✅ |
| CNN Architectures | ✅ |
| Recurrent Neural Networks | ✅ |
| Attention & Transformers | 🔄 In progress |
| Object Detection | |
| Self-Supervised Learning | |

### Bishop — *Deep Learning*

| Chapter | Topic | Status |
|---|---|---|
| 1 | The Deep Learning Revolution | ✅ |
| 2 | Probabilities | ✅ |
| 12 | Transformers | 🔄 In progress |
| 13 | Graph Neural Networks | 🔄 In progress |

---

## Key Implementations

### Image Classification Pipeline

The first block of work covers the core CS231n classification pipeline, implemented from scratch in NumPy on CIFAR-10 (50,000 training images, 10 classes).

- **k-Nearest Neighbour classifier** — implemented in three versions: naive double loop, partial vectorisation, and fully vectorised using broadcasting and L2 distance. Cross-validated over k to find optimal hyperparameters. Useful primarily as a baseline and to build intuition about distance metrics before moving to parametric models.
- **SVM loss (multi-class hinge loss)** — implemented in naive and vectorised forms. Derived and coded the analytic gradient by hand; verified against numerical gradient using finite differences.
- **Softmax loss (cross-entropy)** — implemented alongside SVM for direct comparison. Both share the same linear scoring function; the difference is entirely in how they interpret the scores. Explored why Softmax produces calibrated probabilities while SVM only cares about the margin.
- **Linear classifier trained with SGD** — combined the loss functions with a gradient descent loop; tuned learning rate and L2 regularisation strength via grid search on a validation split.

---

### Neural Networks: Forward & Backward Pass

- **Modular layer design** — implemented `affine_forward`, `affine_backward`, `relu_forward`, `relu_backward` as standalone functions, then composed them into a full network. This modular approach mirrors how autograd frameworks like PyTorch work internally.
- **Two-layer neural network from scratch** — complete forward pass (scores → loss) and backward pass (loss → gradients via chain rule) without any autograd. Gradient checking via numerical Jacobian confirmed correctness before training.
- **Batch Normalisation** — implemented forward pass (normalise → scale/shift) and backward pass. Explored how it smooths the loss landscape and reduces sensitivity to weight initialisation; also implemented the inference-time behaviour using running statistics.
- **Dropout** — implemented inverted dropout (scale at train time, pass through at test time). Verified that disabling dropout at test time produces consistent expectations.

---

### Optimisation

Implemented the following optimisers from scratch and benchmarked convergence on the same problem:

- **Vanilla SGD** — baseline; sensitive to learning rate, slow on flat or ill-conditioned loss surfaces.
- **SGD with Momentum** — accumulates a velocity vector; converges faster on ravines, less sensitive to noisy gradients.
- **RMSProp** — adapts per-parameter learning rates using a running average of squared gradients; better on sparse or noisy gradients.
- **Adam** — combines momentum and RMSProp; includes bias correction for the first few steps. Used as the default optimiser for all PyTorch experiments.

---

### Convolutional Neural Networks

- **Convolution layer — naive implementation** — implemented `conv_forward_naive` and `conv_backward_naive` using explicit loops over batch, channel, height, and width. Slow but correct; used to understand exactly what a convolution computes before moving to optimised implementations.
- **Max pooling** — implemented forward and backward pass. The backward pass requires tracking which position held the max during the forward pass, which is the discrete analogue of the ReLU gradient.
- **CNN in PyTorch** — built and trained a multi-layer CNN on CIFAR-10 using `nn.Conv2d`, `nn.BatchNorm2d`, and `nn.Linear`. Saved trained weights excluded from version control via `.gitignore`.
- **Spatial Batch Normalisation** — extended batch normalisation to handle the (N, C, H, W) tensor format produced by convolutional layers.

---

### CNN Architectures

Studied and replicated in PyTorch the landmark architectures that define the modern deep learning era:

- **AlexNet** — the 2012 ImageNet winner; notable for ReLU activations, dropout regularisation, and GPU training at scale.
- **VGG** — replaced large filters with stacks of 3×3 convolutions to achieve the same receptive field with fewer parameters and more non-linearities.
- **GoogLeNet / Inception** — introduced the Inception module: parallel convolutions at multiple scales concatenated along the channel dimension. Explored why 1×1 convolutions act as a channel-wise fully-connected layer and enable dimensionality reduction.
- **ResNet** — introduced skip (residual) connections to solve the vanishing gradient problem in very deep networks. The key insight: it is easier to learn a residual mapping F(x) than to directly learn the full mapping H(x).

---

### Training Neural Networks

- **Weight initialisation** — compared zero initialisation (broken: symmetry means all neurons learn identically), small random weights (vanishing gradients at depth), Xavier initialisation (variance-preserving for tanh), and He initialisation (variance-preserving for ReLU). Visualised activation distributions and gradient magnitudes across layers to understand why initialisation matters.
- **Activation functions** — implemented and compared sigmoid, tanh, ReLU, Leaky ReLU, and ELU. Analysed the dying ReLU problem and its practical consequences.
- **Learning rate scheduling** — experimented with step decay and cosine annealing.
- **Data augmentation** — applied random crops, horizontal flips, and colour jitter during training; verified improvement in validation accuracy without changing the model architecture.
- **Transfer learning** — fine-tuned a pretrained ResNet backbone on a downstream task by freezing early layers and replacing the final fully-connected head.

---

### Recurrent Neural Networks — Character-Level Language Model

The RNN section is in two parts: a hand-crafted toy demonstration of how RNN memory works, followed by a full character-level language model trained on Shakespeare, based on Andrej Karpathy's min-char-rnn.

**Toy RNN — demonstrating temporal memory**

Before training anything, a small RNN was constructed with hand-set weights to verify intuition about how hidden state carries information forward in time. The network takes a binary input sequence and is wired so that `h[1]` at time `t` receives `h[0]` from time `t-1` — making the output a one-step delayed copy of the input. For input `[1,1,1,1,1,1,1,1,1]` the output is `[0,1,1,1,1,1,1,1,1]`: the first step produces 0 because no prior context exists yet, and every subsequent step produces 1. This is a minimal proof that the hidden state is functioning as memory, before any learning is involved.

**Character-level language model on Shakespeare**

- **Corpus** — `input.txt` contains a Shakespeare excerpt: 632 characters, 42 unique. Small enough to train quickly; rich enough to have real structure (capitalisation, punctuation, poetic metre).
- **Vanilla RNN forward pass** — at each time step, the hidden state is updated as `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)`, where `x_t` is a one-hot character vector. Output logits are computed as `y_t = W_hy @ h_t + b_y`, then passed through softmax to get next-character probabilities. Hidden size: 100 neurons. Sequence length: 25 steps.
- **Backpropagation Through Time (BPTT)** — the backward pass unrolls the RNN for `seq_length` steps and backpropagates through the entire unrolled graph. The tanh gradient is computed explicitly as `dhraw = (1 - h_t²) * dh`, which is the derivative of tanh — not delegated to autograd.
- **Value-based gradient clipping** — gradients are clipped element-wise to `[-5, 5]` before each weight update (`np.clip(dparam, -5, 5)`), preventing weight explosions during unstable training steps.
- **Adagrad optimiser** — weight updates use Adagrad rather than vanilla SGD: each parameter accumulates the sum of squared gradients, and the learning rate is divided by its square root. This means frequently updated weights get smaller effective learning rates over time.
- **Sampling** — after each 100 iterations, the model generates 200 characters by feeding its own output back as the next input. The progression is legible in the training logs: at iteration 0 the output is pure noise; by iteration ~2500 the model produces recognisable words ("compare", "summer", "thee"); by ~8000 it is generating structurally coherent lines including `"Rough winds do shake the darling buds of May"`.
- **Training instability** — at iteration ~17200 the smooth loss spikes from ~0.16 back up to 3.6 and keeps rising briefly before recovering. This is a real instability event visible in the training log, caused by a sequence of inputs that produced large gradients that exceeded what clipping fully controlled. The model recovers and continues converging, which illustrates why monitoring the loss curve matters — a single loss value at the end of training hides the full story.

The Shakespeare corpus is a useful training target precisely because the signal is rich but the corpus is small: the model cannot memorise it blindly, so it is forced to learn structure. The gap between a model at loss ~93 (random character frequencies) and one at loss ~0.16 (near-verbatim recall of the sonnet with occasional drift) is a direct measure of how much temporal structure the RNN has captured over 16,000 iterations.

---

## Running the Notebooks

```bash
# 1. Clone the repository
git clone https://github.com/adityaswamii/deep-learning-for-cv.git
cd deep-learning-for-cv

# 2. Install dependencies
pip install numpy matplotlib jupyter torch torchvision

# 3. Download datasets
# CIFAR-10 and MNIST are loaded automatically by torchvision,
# or via download scripts in lectures/data/ — excluded from version control

# 4. Launch Jupyter
jupyter notebook
```

> **Python version:** 3.10+ recommended. NumPy and Matplotlib are used throughout; PyTorch is required for the CNN and training experiments.

---

## Notes on Approach

The notebooks prioritise understanding over convenience. Where a library function exists (e.g. `torch.nn.Conv2d`), the approach here is to implement it manually first, verify correctness against a numerical gradient check or known-good reference, and only then use the library version. The `/figures` folder contains saved outputs from these experiments.
