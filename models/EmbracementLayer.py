import torch
from torch import nn
import numpy as np
from models.AttentionLayer import AttentionLayer
from models.SelfAttentionLayer import SelfAttention, SelfAttentionPytorch
from pytorch_transformers.modeling_bert import BertAttention, BertSelfAttention

import math


__author__ = "Gwena Cunha"


class EmbracementLayer(nn.Module):
    def __init__(self, config, hidden_size, p, max_seq_length):
        super(EmbracementLayer, self).__init__()
        self.p = p
        self.hidden_size = hidden_size  # 768
        self.max_seq_length = max_seq_length  # 128

        if self.p == 'selfattention':
            self.self_attention = SelfAttention(self.hidden_size)  #self.max_seq_length)  # AttentionLayer(self.hidden_size)
        elif self.p == 'multihead_bertselfattention':
            self.self_attention = BertSelfAttention(config)
        elif self.p == 'multihead_bertattention':
            self.self_attention = BertAttention(config)
        elif self.p == 'multiheadattention':
            config_att = config
            config_att.output_attentions = True
            self.self_attention = BertSelfAttentionScores(config_att)
        elif self.p == 'selfattention_pytorch':
            self.self_attention = SelfAttentionPytorch(self.max_seq_length)  # 128

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
            elif 'multihead_bert' in self.p:
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, attention_mask, head_mask=None)
                selfattention_scores = selfattention_scores[0]
            else:
                # B. Self-attention used to choose most important indexes -> p = softmax(mean(self_att))
                #   'selfattention_scores' shape -> (bs, 128)
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                # ADD THE NEXT 2 LINES TO CONDENSED
                # attention_mask_bs = attention_mask[i_bs, :]
                # _, selfattention_scores = self.self_attention(tokens_to_embrace_bs, attention_mask_bs)
                if self.p == 'selfattention':
                    _, selfattention_scores = self.self_attention(tokens_to_embrace_bs)
                elif self.p == 'multiheadattention':  # BertAttention
                    attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                    tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                    selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, attention_mask, head_mask=None)
                else:  # 'selfattention_pytorch'
                    tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(
                        0).cuda()
                    selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor)
                # Choose features using information from self-attention
                selfattention_scores = torch.tensor(selfattention_scores, dtype=torch.float)
                embraced_features_index = torch.multinomial(selfattention_scores, emb_size, replacement=True)  # shape = [768]
                embraced_features_index = embraced_features_index.cpu().detach().numpy()  # shape = 768

            # 2. Add features into one of size (bs, embedding_size)
            embraced_features_token_bs = []
            if 'multihead_bert' in self.p:
                embraced_features_index = torch.sum(selfattention_scores, dim=1)
                embraced_features_token_bs = embraced_features_index.squeeze()
                embraced_features_token_bs = embraced_features_token_bs.cpu().detach().numpy()
            else:
                for i_emb, e in enumerate(embraced_features_index):
                    embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)  # (768)
        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)  # (bs, 768)

        return embraced_features_token


class BertSelfAttentionScores(nn.Module):
    """ Based on modeling_bert.py > BertSelfAttention """
    def __init__(self, config):
        super(BertSelfAttentionScores, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        #context_layer = torch.matmul(attention_probs, value_layer)
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #context_layer = context_layer.view(*new_context_layer_shape)
        #outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        attention_probs_mean = attention_probs.permute(0, 2, 1, 3).contiguous()
        attention_probs_mean = attention_probs_mean.sum(dim=2)
        outputs = attention_probs_mean.sum(dim=1).squeeze()
        #max_value = torch.max(outputs_tmp).item()
        #outputs = [o/max_value for o in outputs_tmp]
        #outputs = torch.tensor(outputs, dtype=torch.float)
        outputs = nn.Softmax(dim=-1)(outputs)

        return outputs
