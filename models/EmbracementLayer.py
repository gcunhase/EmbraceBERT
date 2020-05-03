import torch
from torch import nn
import numpy as np
from models.AttentionLayer import AttentionLayer
from models.SelfAttentionLayer import SelfAttention, SelfAttentionPytorch,\
    BertSelfAttentionScores, BertSelfAttentionScoresP, BertMultiSelfAttentionScoresP,\
    BertMultiAttentionScoresP, BertAttentionClsQuery
from pytorch_transformers.modeling_bert import BertAttention, BertSelfAttention
from utils import visualize_attention


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
        elif self.p == 'multihead_bertselfattention_in_p':
            config.num_attention_heads = 1
            self.self_attention = BertSelfAttentionScoresP(config)
        elif self.p == 'multihead_bertattention':
            self.self_attention = BertAttention(config)
        elif self.p == 'multihead_bertattention_clsquery':
            self.self_attention = BertAttentionClsQuery(config)
        elif self.p == 'attention_clsquery':
            self.self_attention = AttentionLayer(self.hidden_size)
        elif self.p == 'multiheadattention':
            config_att = config
            config_att.output_attentions = True
            self.self_attention = BertSelfAttentionScores(config_att)
        elif self.p == 'selfattention_pytorch':
            self.self_attention = SelfAttentionPytorch(self.max_seq_length)  # 128
        elif self.p == 'multiple_multihead_bertselfattention_in_p':
            config.num_attention_heads = 1
            self.self_attention = BertMultiSelfAttentionScoresP(config)
        elif self.p == 'multiple_multihead_bertattention_in_p':
            config.num_attention_heads = 1
            config.output_attentions = True
            self.self_attention = BertMultiAttentionScoresP(config, max_seq_length)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, output_tokens_from_bert, cls_token=None):
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
            elif self.p == 'multihead_bertselfattention' or self.p == 'multihead_bertattention':
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, attention_mask, head_mask=None)
                selfattention_scores = selfattention_scores[0]
            elif self.p == 'multihead_bertattention_clsquery':
                print("TODO. Use cls_token - Come back to this")
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                cls_token_bs = cls_token[i_bs, :]
                attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                cls_token_bs = torch.tensor(cls_token_bs, dtype=torch.float).unsqueeze(0).cuda()
                selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, attention_mask, head_mask=None, cls_query=cls_token_bs)
                selfattention_scores = selfattention_scores[0]
            elif self.p == 'attention_clsquery':
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                cls_token_bs = cls_token[i_bs, :]
                tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).cuda()
                cls_token_bs = torch.tensor(cls_token_bs, dtype=torch.float).unsqueeze(0).cuda()
                selfattention_scores = self.self_attention(cls_token_bs, tokens_to_embrace_bs_tensor, unsqueeze_idx=0)
            elif self.p == 'multiple_multihead_bertselfattention_in_p' or self.p == 'multiple_multihead_bertattention_in_p':
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, head_mask=attention_mask,
                                                           is_visualize_attention=False)
                if self.p == 'multiple_multihead_bertattention_in_p':
                    selfattention_scores = selfattention_scores.squeeze()
                    selfattention_scores = self.softmax(selfattention_scores)
                # Choose features using information from self-attention
                multiple_embrace_vectors = []
                for i in range(self.max_seq_length):  # 128
                    score = selfattention_scores[i, :]
                    #attention_probs_img = score.unsqueeze(0).cpu().detach().numpy()
                    #visualize_attention(attention_probs_img)
                    embraced_features_index = torch.multinomial(score, emb_size, replacement=True)  # shape = [768]
                    embraced_features_index = embraced_features_index.cpu().detach().numpy()  # shape = 768
                    embraced_features_token_bs = []
                    for i_emb, e in enumerate(embraced_features_index):
                        embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
                    multiple_embrace_vectors.append(embraced_features_token_bs)
                multiple_embrace_vectors = torch.tensor(multiple_embrace_vectors, dtype=torch.float)
            else:
                # B. Self-attention used to choose most important indexes -> p = softmax(mean(self_att))
                #   'selfattention_scores' shape -> (bs, 128)
                tokens_to_embrace_bs = tokens_to_embrace[i_bs, :, :]
                # ADD THE NEXT 2 LINES TO CONDENSED
                # attention_mask_bs = attention_mask[i_bs, :]
                # _, selfattention_scores = self.self_attention(tokens_to_embrace_bs, attention_mask_bs)

                # Original attention_mask ranges from 0 to -1000
                #    If we want to mask the scores by multiplying between 0 and 1, we should give the attention_mask
                #      as head_mask
                if self.p == 'selfattention':
                    _, selfattention_scores = self.self_attention(tokens_to_embrace_bs)
                elif self.p == 'multiheadattention':  # BertAttention
                    attention_mask = torch.ones([1, 1, 1, np.shape(tokens_to_embrace_bs)[0]]).cuda()
                    tokens_to_embrace_bs_tensor = torch.tensor(tokens_to_embrace_bs, dtype=torch.float).unsqueeze(0).cuda()
                    selfattention_scores = self.self_attention(tokens_to_embrace_bs_tensor, attention_mask, head_mask=None)
                elif self.p == 'multihead_bertselfattention_in_p':
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
            if self.p == 'multihead_bertselfattention' or self.p == 'multihead_bertattention':
                embraced_features_index = torch.sum(selfattention_scores, dim=1)
                embraced_features_token_bs = embraced_features_index.squeeze()
                embraced_features_token_bs = embraced_features_token_bs.cpu().detach().numpy()
            elif self.p == 'multiple_multihead_bertselfattention_in_p' or self.p == 'multiple_multihead_bertattention_in_p':
                embraced_features_token_bs = torch.sum(multiple_embrace_vectors, dim=0)
                embraced_features_token_bs = embraced_features_token_bs.squeeze()
                embraced_features_token_bs = embraced_features_token_bs.cpu().detach().numpy()
            elif self.p == 'attention_clsquery':
                embraced_features_token_bs = selfattention_scores.cpu().detach().numpy()
            else:
                for i_emb, e in enumerate(embraced_features_index):
                    embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)  # (768)
        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)  # (bs, 768)

        return embraced_features_token
