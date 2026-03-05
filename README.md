# Reward-Based Image Detection using Inverse Reinforcement Learning

A research-focused implementation of **Inverse Reinforcement Learning (IRL)** for object detection.  
This repository explores how bounding box prediction can be formulated as a **reward-driven decision-making problem** rather than purely supervised regression.

Instead of directly minimizing bounding box regression loss, we learn a **reward function** that explains expert annotations and train a detection policy to maximize that learned reward.

---

## 🧠 Motivation

-> Traditional object detection:

image → neural network → bounding box
optimize: regression loss (e.g., Smooth L1, IoU)

-> Reward-based detection (this project):

image → feature representation
learn reward R(s, a) from expert demonstrations
train detection policy to maximize R(s, a)


This separation between:
- **What is good?** (reward learning)
- **How to act?** (policy learning)

enables more flexible, interpretable, and human-aligned learning.

---

# 📂 Repository Structure

```
ayoub-elkhaiari-reward_based-image-detection-inverse-reinforcement-learning-init-/
│
├── IRL_final/
│ ├── main.py
│ ├── models/
│ │ ├── detection_model.py
│ │ └── reward_model.py
│ └── utils/
│ ├── dataset.py
│ ├── evaluation.py
│ └── train.py
│
├── IRL_GAIL/
│ ├── main.py
│ ├── models/
│ │ ├── detection_model.py
│ │ └── reward_network.py
│ └── utils/
│ ├── dataset.py
│ ├── evaluation.py
│ └── train.py
│
└── IRL_MaxEntropy/
├── main.py
├── models/
│ ├── detection_model.py
│ └── reward_network.py
└── utils/
├── dataset.py
├── evaluation.py
└── train.py
```

Each module contains:

- `main.py` → Entry point  
- `models/` → Detection and reward architectures  
- `utils/dataset.py` → Data loading & preprocessing  
- `utils/train.py` → Training loop  
- `utils/evaluation.py` → IoU and Dice metrics  

---

# 🔬 Implementations

This repository contains three IRL-based detection strategies.

---

## 1️⃣ IRL_MaxEntropy

Implements **Maximum Entropy Inverse Reinforcement Learning**.

### Core Idea

Learn a reward function \( R_\theta(s, a) \) such that expert bounding boxes are exponentially more likely than alternatives:

\[
P(a|s) \propto \exp(R_\theta(s, a))
\]

### Characteristics

- Probabilistic formulation
- Encourages robust stochastic behavior
- Theoretically grounded
- Stable reward estimation

Best suited for:
- Research experiments
- Controlled reward modeling
- Likelihood-based learning

---

## 2️⃣ IRL_GAIL

Implements **Generative Adversarial Imitation Learning (GAIL)**.

### Core Idea

- Detection model = Generator  
- Reward network = Discriminator  
- Expert vs predicted bounding boxes compete adversarially  

### Characteristics

- Implicit reward modeling
- No explicit partition function
- Learns complex expert distributions
- Can model subtle annotation biases

Best suited for:
- Adversarial imitation learning experiments
- Complex reward landscapes

---

## 3️⃣ IRL_final

Refined and stabilized version integrating:

- Improved detection head
- Cleaner reward-policy interaction
- More stable optimization
- Modular training structure

Recommended for:
- Final experiments
- Benchmarking
- Production-style experiments

---

# 🏗 Model Components

## Detection Model

- CNN-based feature extractor
- Fully connected bounding box head
- Outputs bounding box coordinates

## Reward Model

- Takes (state, action) as input
- Outputs scalar reward
- Guides policy optimization

---

# 📊 Evaluation Metrics

Implemented metrics:

- **IoU (Intersection over Union)**
- **Dice Score**

These metrics measure spatial overlap between predicted and expert bounding boxes.

---

# ⚙️ How to Run

### Run Maximum Entropy IRL

```bash
cd IRL_MaxEntropy
python main.py
```
## Run GAIL-based IRL

```bash
cd IRL_GAIL
python main.py
```
## Run GAIL-based IRL

```bash
cd IRL_GAIL
python main.py
```

## Run GAIL-based IRL
```bash
cd IRL_final
python main.py
```

# 🧪 Training Workflow

Each implementation follows this pipeline:

Load dataset

Extract state representations

Learn reward function

Update detection policy

Evaluate with IoU/Dice

# 🖥 Computational Notes

On CPU:

Reduce batch size

Reduce action sampling

Use smaller backbone (e.g., ResNet18)

On GPU:

Increase sampling

Enable full joint optimization

Use larger batch sizes

# 🎯 Research Significance

This project explores:

Inverse Reinforcement Learning for vision

Reward modeling for structured prediction

Human-aligned AI systems

Evaluation-aware learning

It demonstrates how object detection can be framed as:

Learning what is valuable, rather than directly predicting coordinates.


---

## 🧮 Convex Optimization-Based Reward Learning

In addition to adversarial and entropy-based IRL methods, this repository includes a **convex max-margin formulation** for learning reward functions.

This approach reformulates reward learning as a **structured convex optimization problem**, inspired by structured SVMs and margin-based learning.

---

### 📂 File
Convex_optimization.ipynb

---

## 🧠 Core Idea

Instead of learning the reward function through gradient-based neural optimization, we solve a **convex optimization problem** that enforces expert bounding boxes to score higher than non-expert candidates.

For each image:

1. Extract deep features using a pretrained ResNet-50 backbone.
2. Generate candidate bounding boxes via grid sampling.
3. Compute feature difference vectors:

\[
\phi(s, a_{expert}) - \phi(s, a_{candidate})
\]

4. Solve the following convex problem:

\[
\min_\theta \frac{1}{2} \|\theta\|^2 + C \sum_i \xi_i
\]

Subject to:

\[
X\theta \ge 1 - \xi, \quad \xi \ge 0
\]

Where:
- \( \theta \) = reward parameter vector  
- \( X \) = matrix of feature differences  
- \( \xi \) = slack variables  
- \( C \) = regularization parameter  

This ensures expert actions receive higher reward than alternatives with a margin.

---

## 🔬 Interpretation

This formulation is equivalent to a **structured max-margin reward learning problem**, where:

- The reward function is linear:  
  \[
  R(s, a) = \theta^T \phi(s, a)
  \]

- The optimal solution is globally optimal due to convexity.
- No adversarial instability.
- No partition function approximation.

---

## 📊 Pipeline

1. Load annotated dataset  
2. Extract CNN features for expert and candidate boxes  
3. Build feature difference matrix  
4. Solve convex optimization using CVXPY  
5. Predict bounding box by maximizing learned reward  
6. Evaluate using IoU and Dice  

---

## ⚖ Comparison with Other Methods

| Method | Optimization Type | Stability | Interpretability |
|--------|-------------------|-----------|------------------|
| MaxEntropy IRL | Probabilistic | Medium | Medium |
| GAIL | Adversarial | Lower | Low |
| Final IRL | Hybrid | High | Medium |
| **Convex Optimization** | Convex (Global Optimum) | **Very High** | **High** |

---

## 🚀 Advantages

- Guaranteed global optimum  
- High interpretability of reward weights  
- Theoretically grounded in convex optimization  
- Computationally transparent  
- Strong connection to structured prediction theory  

---

## 🎯 Research Insight

This module bridges:

- Inverse Reinforcement Learning  
- Structured SVMs  
- Margin-based reward modeling  
- Vision-based decision systems  

It demonstrates that reward learning for object detection can be solved without adversarial training or probabilistic normalization, offering a stable and mathematically principled alternative.

---
