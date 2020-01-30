import torch
from torch import nn
import numpy as np
from models.AttentionLayer import AttentionLayer

__author__ = "Gwena Cunha"


class EmbracementLayer(nn.Module):
    def __init__(self, hidden_size):
        super(EmbracementLayer, self).__init__()
        self.embrace_attention = AttentionLayer(hidden_size)

    def forward(self, output_tokens_from_bert):
        # pooled_enc_output = bs x 768
        # output_tokens_from_bert = bert_output[0]
        # cls_output = bert_output[1]  # CLS

        # Note: Docking layer not needed given that all features have the same size
        tokens_to_embrace = output_tokens_from_bert[:, 1:, :]  # (8, 128, 768) = (bs, sequence_length (where the first index is CLS), embedding_size)
        [bs, seq_len, emb_size] = tokens_to_embrace.size()
        tokens_to_embrace = tokens_to_embrace.cpu().detach().numpy()

        # Note: Consider each token in the sequence of 128 as one modality.
        # 1. Multinomial distribution: randomly chose features from all 128 with same probability for each index feature
        probability = torch.tensor(np.ones(seq_len), dtype=torch.float)
        embraced_features_index = torch.multinomial(probability, emb_size, replacement=True)
        embraced_features_index = embraced_features_index.cpu().detach().numpy()

        # 2. Add features into one of size (bs, embedding_size)
        embraced_features_token = []
        for i_bs in range(bs):
            embraced_features_token_bs = []
            for i_emb, e in enumerate(embraced_features_index):
                embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)
        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)

        return embraced_features_token
