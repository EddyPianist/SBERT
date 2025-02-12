#turning dataset into format that can fit sentence transformer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch


# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#customized multi-tasks dataset loader. The original dataset is used for classification, to train the real world multi-tasks
#model, the dataset should has labels for both tasks. For the second task Sentence Similarity detection, we simply set
#three different class Entailment, Contradiction, Neutral as 1, -1, 0 seperately. 
class SNLIDataset(Dataset):
    def __init__(self, dataset):
        """
        dataset: An iterable of SNLI-like dicts 
                 (each has keys: 'premise', 'hypothesis', 'label')
        """
        # Define label mapping for comparing cosine similarity later (map Entailment, Contradiction, Neutral to 2, 1, 0)
        self.label_map = {2: 0, 1: 1, 0: 2}
        
        self.data = [
            (
                example["premise"], 
                example["hypothesis"], 
                self.label_map[example["label"]]
            )
            for example in dataset
            if example["label"] != -1
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each item: (premise_text, hypothesis_text, mapped_label)
        return self.data[idx]


def collate_fn(batch):


    # batch is a list of tuples: [(premise, hypothesis, label), (...), ...]
    premises, hypotheses, labels = zip(*batch)  # each is a tuple of size batch_size

    # Tokenize premises
    premise_encodings = tokenizer(
        list(premises),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Tokenize hypotheses
    hypothesis_encodings = tokenizer(
        list(hypotheses),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    sentence_features = [
        premise_encodings,     # (dict with input_ids, attention_mask, etc.)
        hypothesis_encodings 
    ]
    
    # Convert labels to a single tensor, shape [batch_size]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "sentence_features": sentence_features,
        "label": labels_tensor
    }


def create_loader(dataset, batch_size=16, shuffle=True):
    """
    Wrap SNLIDataset in a DataLoader using the custom collate_fn.
    """
    dataset = SNLIDataset(dataset)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )
