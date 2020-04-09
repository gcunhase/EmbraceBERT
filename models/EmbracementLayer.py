import torch
from torch import nn
import numpy as np
from models.AttentionLayer import AttentionLayer
from models.SelfAttentionLayer import SelfAttention

__author__ = "Gwena Cunha"


class EmbracementLayer(nn.Module):
    def __init__(self, hidden_size, p):
        super(EmbracementLayer, self).__init__()
        self.p = p
        self.hidden_size = hidden_size

        if self.p == 'selfattention':
            self.self_attention = SelfAttention(self.hidden_size)  #self.max_seq_length)  # AttentionLayer(self.hidden_size)

    def forward(self, output_tokens_from_bert):
        # pooled_enc_output = bs x 768
        # output_tokens_from_bert = bert_output[0]
        # cls_output = bert_output[1]  # CLS

        # Note: Docking layer not needed given that all features have the same size
        # tokens_to_embrace = output_tokens_from_bert[:, 1:, :]  # (8, 128, 768) = (bs, sequence_length (where the first index is CLS), embedding_size)
        tokens_to_embrace = output_tokens_from_bert[:, :, :]  # (8, 128, 768) = (bs, sequence_length, embedding_size)
        [bs, seq_len, emb_size] = tokens_to_embrace.size()
        tokens_to_embrace = tokens_to_embrace.cpu().detach().numpy()

        # Note: Consider each token in the sequence of 128 as one modality.
        embraced_features_token = []
        for i_bs in range(bs):
            # 1. Choose feature indexes to be considered in the Embrace vector
            if self.p == 'multinomial':
                # A. Multinomial distribution: randomly choose features from all 128 with same probability for each index feature
                probability = torch.tensor(np.ones(seq_len), dtype=torch.float)
                embraced_features_index = torch.multinomial(probability, emb_size, replacement=True)  # shape = [768]
                embraced_features_index = embraced_features_index.cpu().detach().numpy()  # shape = 768
            else:
                # B. Self-attention used to choose most important indexes -> p = softmax(mean(self_att))
                #   'selfattention_scores' shape -> (bs, 128)
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                # ADD THE NEXT 2 LINES TO CONDENSED
                # attention_mask_bs = attention_mask[i_bs, :]
                # _, selfattention_scores = self.self_attention(tokens_to_embrace_bs, attention_mask_bs)
                _, selfattention_scores = self.self_attention(tokens_to_embrace_bs)
                # Choose features using information from self-attention
                selfattention_scores = torch.tensor(selfattention_scores, dtype=torch.float)
                embraced_features_index = torch.multinomial(selfattention_scores, emb_size, replacement=True)  # shape = [768]
                embraced_features_index = embraced_features_index.cpu().detach().numpy()  # shape = 768

            # 2. Add features into one of size (bs, embedding_size)
            embraced_features_token_bs = []
            for i_emb, e in enumerate(embraced_features_index):
                embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)
        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)

        return embraced_features_token
