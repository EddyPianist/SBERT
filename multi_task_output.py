import torch
import torch.nn as nn
from torch.nn import functional as F
from collections.abc import Iterable
from torch import Tensor, nn


###TASK TWO: Multi-Task Learning Expansion

#task1: Classification task2: Sentence Similarity
#Use two loss function and do a weighted sum as the final loss
#The implement is inspired by SBERT: https://arxiv.org/abs/1908.10084
class Multi_task_output(nn.Module):
    def __init__(
        self,
        model,
        sentence_embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = True,
        concatenation_sent_difference: bool = True,
        concatenation_sent_multiplication: bool = False,
    )->None:
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep                         
        self.concatenation_sent_difference = concatenation_sent_difference          #set true if we include difference two sentence embeddings (default true)
        self.concatenation_sent_multiplication = concatenation_sent_multiplication  #set true if we include multiplication two sentence embeddings (default false)
        self.loss_fct_1 = nn.MSELoss()
        self.loss_fct_2 = nn.CrossEntropyLoss()
        self.alpha = 0.5


        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        self.classifier = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, num_labels
        )

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        reps = [self.model(sentence_feature) for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        
    
        cos_sim = F.cosine_similarity(rep_a, rep_b)                             #calculate cosine similarity
        MSE_loss = self.loss_fct_1(cos_sim, labels.float().view(-1) - 1)        #calculate the loss between predicted similarity and true similarity       

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        
        if labels is not None:                                                  #for model training
            Softmax_loss = self.loss_fct_2(output, labels.view(-1))

            loss = self.alpha * Softmax_loss + (1 - self.alpha) * MSE_loss
            return reps, loss
        else:                                                                   #for inference
            return reps, output, cos_sim
