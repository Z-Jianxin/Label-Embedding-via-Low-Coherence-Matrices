# PyTorch Neural Network Implementation of "Label Embedding via Low-Coherence Matrices"

## Environment Setup

The environment setup is stored in the `env.yml` file. 

## Code Structure

Our codebase includes various Python scripts, with each file serving a different purpose. Here's a brief rundown:

- `nn_utils.py`: This file contains utility functions used in our neural network models.
- `run_nn_[dataset].py`: These scripts implement the embedding methods for different datasets. Replace `[dataset]` with the name of the dataset you are working with (e.g., `lshtc1`, `dmoz`, `odp`).

## Running the Code

To run the embedding framework scripts, use the following command:

```bash
python run_nn_[dataset].py <embedding dimension> <embedding type> <epochs to train> <random seed>
```

Replace `[dataset]` with the name of the dataset you are working with, and `<embedding dimension>` with the desired embedding dimension. ` <embedding type>` is the type of embedding, which can be `rademacher`, `gaussian`, `gaussian_complex`, `nelson_complex`. For example:

```bash
python run_nn_lshtc1.py 64 rademacher 5 86774275
```

```

## Data

Our data can be downloaded from this link: [https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing](https://drive.google.com/drive/folders/1K3lU4vhHTFYrZs2L5sY9uC3gRGeft5aS?usp=sharing).
