# Sentence BERT: Multi-Task Model Implementation

This project implements Sentence-BERT (SBERT) and modifies it into a multi-task model. You can refer to Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/pdf/1810.04805 for more information. 

## Installation

### Step 1: Create a Conda Environment
Run the following command to create a Conda environment with Python 3.8:
```sh
conda create -n myenv python=3.8
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

## Task 1: Sentence BERT Implementation
- The implementation of Sentence BERT can be found in `BERT.py`.
- The model structure is similar to the original Transformer but includes an additional average pooling layer at the end.
- This pooling layer converts variable-length token embeddings into a single sentence embedding while maintaining the same dimension.
- The hyperparameters follow those used in [this paper](https://arxiv.org/pdf/1810.04805) on BERT-based models.

## Task 2: Multi-Task Model Implementation
- The code for this task can be found in `multi_task_output.py`.
- The forward pass is modified to include two objective functions:
  1. **Classification Objective Function**: Sentence embeddings `u` and `v` are concatenated along with their element-wise absolute difference `|u − v|`. The resulting vector is then multiplied with a trainable weight.
  2. **Regression Objective Function**: Computes the cosine similarity between sentence embeddings `u` and `v`, using mean-squared error (MSE) loss as the objective function.

## Task 3: Transfer Learning and Model Freezing Strategies
We consider different scenarios for freezing parts of the network:
1. **Freezing the entire network**: Used for evaluation purposes, ensuring model weights remain unchanged.
2. **Freezing only the Transformer backbone**: Allows fine-tuning of task-specific layers while keeping the base model stable. In our specific senario, the cosine similarity fully depended on the sentence embeddings from the backbone, so if we find out the loss of the Regression Objective Function is low while Classification Objective Function is high, we can use this strategy. 
3. **Freezing only one of the task-specific heads**: This approach enables transfer learning where one task remains fixed while the other adapts to new data. For our model, we only have the classification head thus this strategy might not be applied. But in general, if the model is good at one task but not at the other, we can use this training strategy. 

### Transfer Learning Strategy
- **Selecting a pre-trained model**: Transfer learning can benefit our work a lot because there are a lot of open-source pre-trained transformer models for language tasks in various sizes. The choice depends on dataset size and task complexity. For this demo, the structure of the model is the same as BERT-base, as the result, we can load the pre-trained BERT-base weights to our model. 
- **Freezing layers**:
  - Freezing all layers except task heads is efficient, it is the first choice if the compuational resoure is low.
  - Freezing most layers except the last few provides a balance between efficiency and flexibility.
  - Fine-tuning the entire model yields the best results but requires comparatively great amount of computational resources.

## Task 4: Training and Data Loading
- The training process is implemented in `train.py`.
- A small dataset is loaded using a customized data loader for demonstration purposes.

## Summary
This project provides a modular implementation of Sentence-BERT with a multi-task objective. The model supports classification and regression tasks while allowing different transfer learning strategies to improve performance. The training setup ensures flexibility for various applications.

For further details, please refer to the respective Python files in the repository.

