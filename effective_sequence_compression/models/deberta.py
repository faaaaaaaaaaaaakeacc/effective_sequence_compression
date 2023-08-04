import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import *


class DebertaV2EncoderModified(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, encoder, truncation_arr):
        super().__init__()
        self.layer = encoder.layer
        self.relative_attention = encoder.relative_attention
        if self.relative_attention:
            self.max_relative_positions = encoder.max_relative_positions
            self.position_buckets = encoder.position_buckets
            self.rel_embeddings = encoder.rel_embeddings
        self.norm_rel_ebd = encoder.norm_rel_ebd
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = encoder.LayerNorm
        self.conv = encoder.conv
        self.gradient_checkpointing = encoder.gradient_checkpointing
        self.truncation_arr = truncation_arr

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):  
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                output_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class DebertaWrapper(nn.Module):
    def __init__(self, n_classes, truncation_list = None):
        super().__init__()
        self.truncation_list = truncation_list
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
        self.classifier = nn.Linear(384, n_classes)

    def forward(self, input_texts):
        input_batch = self.tokenizer(input_texts, return_tensors="pt")
        if self.truncation_list is None:
            output = self.model(**input_batch).last_hidden_state
            output = torch.mean(output, dim=1)
            return self.classifier(output)
        else:
            input_tokens = input_batch['input_ids']
            embeds = self.model.embeddings(input_tokens)
            total_length = embeds.shape[1]
            for i in range(len(self.truncation_list)):
                output = self.model.encoder.layer[i](embeds, torch.ones_like(embeds))
                cur_length = int(self.truncation_list[i] * total_length)
                output = output[:, :cur_length, :]
            output = torch.mean(output, dim=1)
            return self.classifier(output)
