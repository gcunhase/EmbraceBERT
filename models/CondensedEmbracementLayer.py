import torch
from torch import nn
import numpy as np
from models.AttentionLayer import AttentionLayer

__author__ = "Gwena Cunha"


class CondensedEmbracementLayer(nn.Module):
    def __init__(self, hidden_size):
        super(CondensedEmbracementLayer, self).__init__()
        self.embrace_attention = AttentionLayer(hidden_size)

    def forward(self, bert_output, attention_mask):
        # pooled_enc_output = bs x 768
        output_tokens_from_bert = bert_output[0]
        cls_output = bert_output[1]  # CLS

        # Note: Docking layer not needed given that all features have the same size
        [bs, emb_size] = cls_output.size()
        embraced_features_token = []
        for i_bs in range(bs):
            # 0. Obtain relevant tokens (Embracement layer with outputs between CLS and SEP only)
            attention_i_bs = attention_mask[i_bs].cpu().detach().numpy()
            last_idx = 0
            for i_att, att in enumerate(attention_i_bs):
                if att == 1:
                    last_idx = i_att
                else:
                    break
            tokens_to_embrace = output_tokens_from_bert[i_bs, 0:last_idx, :]

            [seq_len, emb_size] = tokens_to_embrace.size()
            tokens_to_embrace = tokens_to_embrace.cpu().detach().numpy()

            # Note: Consider each token in the sequence of 128 as one modality.
            # 1. Multinomial distribution: randomly chose features from all 128 with same probability for each index feature
            probability = torch.tensor(np.ones(seq_len), dtype=torch.float)
            embraced_features_index = torch.multinomial(probability, emb_size, replacement=True)
            embraced_features_index = embraced_features_index.cpu().detach().numpy()

            # 2. Add features into one of size (bs, embedding_size)
            embraced_features_token_bs = []
            for i_emb, e in enumerate(embraced_features_index):
                embraced_features_token_bs.append(tokens_to_embrace[e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)

        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)

        # 3. Apply attention layer to CLS and embraced_features_token
        embrace_output = self.embrace_attention(cls_output, embraced_features_token)

        return embrace_output
