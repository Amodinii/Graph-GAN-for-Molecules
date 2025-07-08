# GraphGAN for Molecular Generation

## Overview

This repository implements a custom Generative Adversarial Network (GAN) for molecular graph generation. The objective is to generate novel, chemically valid molecules by learning from existing molecular data. Molecules are represented as graphs, with atoms as nodes and chemical bonds as edges. The model is designed to assist in tasks like drug discovery and chemical space exploration by proposing new molecular structures that resemble real compounds.

## Model Architecture

The architecture follows a standard GAN setup with domain-specific adaptations for molecular graphs:

- **Generator**: Produces molecular graphs in the form of an adjacency matrix (representing bonds) and a node feature matrix (representing atom types).
- **Discriminator**: Takes a molecular graph as input and determines whether it is real (from the dataset) or fake (generated).

Both networks are trained in an adversarial loop where the generator tries to fool the discriminator, and the discriminator learns to distinguish generated molecules from real ones.

## Dataset

The model is trained on the **QM9** dataset, which contains approximately 134,000 small organic molecules with up to nine heavy atoms (C, O, N, F).

### Preprocessing

Molecules are preprocessed by converting them into graph representations:
- **Adjacency matrices** to capture the molecular bonding structure.
- **Feature matrices** to encode atom-level features such as atom type.

These graph representations serve as inputs for both training and evaluation.

## Training and Evaluation

The training loop alternates between:
- Updating the discriminator to improve its ability to classify real and generated graphs.
- Updating the generator to produce molecular graphs that can successfully fool the discriminator.
