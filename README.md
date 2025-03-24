# Sentence BERT: Multi-Task Model Implementation

This project implements Sentence-BERT (SBERT) and modifies it into a multi-task model. You can refer to Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/pdf/1810.04805 for more information. 

## Installation

### Step 1: Create a Conda Environment
Run the following command to create a Conda environment with Python 3.10:
```sh
conda create -n myenv python=3.10
```

### Step 2: Install Dependencies
After creating the environment, install the required dependencies by running:
```sh
pip install -r requirements.txt
```

### Step 3: Train the Model
Once all packages are installed, you can start training the model using:
```sh
python train.py
```

## Part 1: Sentence BERT Implementation
- The implementation of Sentence BERT can be found in `BERT.py`.
- The model structure is similar to the original Transformer but includes an additional average pooling layer at the end.
- This pooling layer converts variable-length token embeddings into a single sentence embedding while maintaining the same dimension.
- The hyperparameters follow those used in [this paper](https://arxiv.org/pdf/1810.04805) on BERT-based models.

## Part 2: Multi-Task Model Implementation
- The code for this part can be found in `multi_task_output.py`.
- The forward pass is modified to include two objective functions:
  1. **Classification Objective Function**: Sentence embeddings `u` and `v` are concatenated along with their element-wise absolute difference `|u − v|`. The resulting vector is then multiplied with a trainable weight.
  2. **Regression Objective Function**: Computes the cosine similarity between sentence embeddings `u` and `v`, using mean-squared error (MSE) loss as the objective function.


## Part 3: Training Loop Implementation
- The training process is implemented in `train.py`.
- A small dataset is loaded using a customized data loader for demonstration purposes.
- The training supposes to take less than 1 min because only 1000 samples are used for training. 

## Summary
This project provides a modular implementation of Sentence-BERT with a multi-task objective. The model supports classification and regression tasks while allowing different transfer learning strategies to improve performance. The training setup ensures flexibility for various applications.

For further details, please refer to the respective Python files in the repository.

