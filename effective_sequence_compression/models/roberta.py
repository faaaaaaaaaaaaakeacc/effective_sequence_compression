import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import *


class RobertaWrapper(nn.Module):
    def __init__(self, device, n_classes, truncation_list = None):
        super().__init__()
        self.truncation_list = truncation_list
        self.model = AutoModel.from_pretrained("roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.classifier = nn.Linear(768, n_classes)
        self.device = device

    def forward(self, input_texts):
        input_batch = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        input_batch = {key: tensor.to(self.device) for key, tensor in input_batch.items()}
        if self.truncation_list is None:
            output = self.model(**input_batch).last_hidden_state
            output = torch.mean(output, dim=1)
            return self.classifier(output)
        else:
            input_tokens = input_batch['input_ids']
            embeds = self.model.embeddings(input_tokens)
            total_length = embeds.shape[1]
            for i in range(len(self.truncation_list)):
                output = self.model.encoder.layer[i](embeds)[0]
                cur_length = max(1, int(self.truncation_list[i] * total_length))
                output = output[:, :cur_length, :]
            output = torch.mean(output, dim=1)
            return self.classifier(output)
