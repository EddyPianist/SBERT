# SBERT
 Implement the Sentence BERT and modify it into a multi-task model

 ##Task 1:

The code can be found in BERT.py. The model structure is almost the same as original transformer structure except that there is a average pooling layer in the end which transfer multiple verious-length token embeddings into one sentence embedding. (The dimension remains the same.) The hyperparams are exactly the same as BERT-based in this work: https://arxiv.org/pdf/1810.04805. 

 ##Task 2:
The code can be found in multi_task_output.py. Modify the forward path of the original Sentence which concludes two objective function: 1. Classification Objective Function:  concatenate the sentence embeddings u and v with the element-wise difference |u−v| and multiply it with the trainable weight. 2. Regression Objective Function: The cosinesimilarity between the two sentence embeddings u and v is computed, then use mean-squared-error loss as the objective function. 

 ##Task 3:
 1. If the entire network should be frozen: we do evaluation in this senario. Because the parameters of the network will not change at this setting, we can eval how good is our model without worrying loss the current weight.
 2. If only the transformer backbone should be frozen:
 3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
    Transfer learning can benefit our work a lot because there are a lot of open-source pre-trained transformer models for language tasks in various sizes. To enable transfer learning, firstly we need to decide the size of our model (depends on the size of our dataset, the complexity of the problem we got to solve.) After we select a pre-trained model, depending on the computation resource, we can choose from frozen all the layer except our task heads, frozen most of the layers except last few layers or frozen none of them. In general, frozen most of the layers except last few layers would be a good choice which is both efficient and effective, but if we have unlimited computational resource, fine-tuned the whole network can give us a very good result.

##Task 4:
 Code can be found at train.py. A small amount of data is loaded by customized dataloader for demo purpose. 
