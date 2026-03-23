# primitive-neural-network

This repository implements the core building blocks of modern neural networks **from scratch**, without relying on frameworks like PyTorch.

The objective is not performance, but **understanding** — to reconstruct the fundamental invariants behind learning systems and build a solid foundation for future architectural experimentation.

## Philosophy

Modern deep learning frameworks abstract away critical mechanics. This repo intentionally removes those abstractions to expose:

- How gradients actually flow
- How parameters update through backpropagation
- How neural networks emerge from simple composable functions

The goal is to move from **"using models" → "understanding systems" → "inventing architectures"**

## Components

### `autograd-scalar`

Implements a minimal automatic differentiation engine at the **scalar level**.

Focus:
- Reverse-mode autodiff (backpropagation)
- Computational graph construction
- Local gradients → global gradient propagation

Key Insight:
> Every gradient can be expressed as:  
> **global gradient × local derivative**

Planned Extensions:
- Additional activation functions
- Gradient flow visualization
- Numerical gradient checking

### `autograd-tensor`

Extends the scalar autograd system to **tensor-based operations**, which are required for practical neural networks.

Focus:
- Vectorized operations
- Broadcasting rules
- Efficient gradient propagation across dimensions

This bridges the gap between conceptual understanding (scalar) and real-world computation (tensor).

### `network`

Builds neural networks on top of the autograd engine.

Focus:
- Layer abstractions (Linear, Activation)
- Forward / backward passes
- Loss functions
- Training loops

Goal:
- Train small neural networks on small datasets
- Validate correctness of the autograd system
- Observe learning dynamics directly

## Roadmap

- [ ] Scalar autograd (complete)
- [ ] Tensor autograd
- [ ] MLP implementation
- [ ] Training on toy datasets (e.g. XOR, MNIST subset)
- [ ] Experiment with custom architectures

## Why this matters

This repository is a **long-term compounding artifact**.

It serves as:
- A personal ground-truth reference for neural network mechanics
- A sandbox for experimenting with new ideas
- A foundation for future systems beyond standard architectures

## Guiding Principle

> If you can't build it from scratch, you don't truly understand it.