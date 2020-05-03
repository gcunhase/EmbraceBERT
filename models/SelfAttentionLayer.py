import torch
from torch import nn
from torch.nn import Parameter, init
from torch.autograd import Variable
import numpy as np
from torchnlp.nn.attention import Attention
import math
import matplotlib.pyplot as plt
from utils import visualize_attention
from pytorch_transformers.modeling_bert import BertLayerNorm, prune_linear_layer


class SelfAttention(nn.Module):
    """Code heavily based on https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4
       Output: scores, Tensor with shape [128]

       WRONG? Output should be a matrix 128x128 -> Words x Words
    """
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size)) #(768, 128)
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        # inputs = np.array(inputs)  #.detach().numpy()  # (bs, 128, 768)
        mat_mul = np.matmul(inputs, self.attention_weights.cpu().detach().numpy())  # att_weights = [768]
        mat_mul = torch.tensor(mat_mul, dtype=torch.float)
        scores = self.non_linearity(mat_mul)  # (bs, 128)
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        # attention_mask is already in the correct format
        if attention_mask is not None:
            mask = torch.tensor(attention_mask, dtype=torch.float)

            # apply the mask - zero out masked timesteps
            masked_scores = scores * mask  # (bs, 128)

            # re-normalize the masked scores
            _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
            scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        inputs_tensor = torch.tensor(inputs, dtype=torch.float)
        weighted = torch.mul(inputs_tensor, scores.unsqueeze(-1).expand_as(inputs_tensor))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


class SelfAttentionPytorch(nn.Module):
    """ Compare each word in query with context

        Tokens to embrace = Context = [bs, 128, 768]
        Query=[bs, 1, 768]

        WRONG? Output should be a matrix 128x128 -> Words x Words
    """
    def __init__(self, hidden_size):
        super(SelfAttentionPytorch, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = Attention(self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens_to_embrace_bs):  #query, context):
        context = tokens_to_embrace_bs.permute(0, 2, 1).contiguous()
        context_size = context.size()
        selfattention_scores = torch.zeros([context_size[0], 1, context_size[2]])
        for i in range(0, context_size[1]):
            query = context[:, i, :]
            query = query.unsqueeze(1)
            attention_score_word, _ = self.attention_layer(query, context)
            selfattention_scores = torch.add(selfattention_scores, attention_score_word.squeeze().cpu())

        selfattention_scores = selfattention_scores / selfattention_scores.max()

        selfattention_scores = self.softmax(selfattention_scores)
        selfattention_scores = selfattention_scores.squeeze().squeeze()
        return selfattention_scores


class BertSelfAttentionScores(nn.Module):
    """ Gwena's modification
        Based on modeling_bert.py > BertSelfAttention

        The original output (attention_probs) has shape [1, 12, 128, 128].
        This output has shape [1, 128]:
            1. Permutation (switch axis 1 and 2) -> [1, 128, 12, 128]
            2. Sum dim=3 -> [1, 128, 12, 1]
            3. Sum dim=2 -> [1, 128, 1, 1]
            4. Squeeze -> [1, 128]
            Drawback: no significance for these actions
    """
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


class BertSelfAttentionScoresP(nn.Module):
    """ Gwena's modification

        Based on modeling_bert.py > BertSelfAttention

        Assumes N=1 (num_attention_heads)
        Visualize AttentionScores Matrix

        The original output (attention_probs) has shape [1, N, 128, 128].
        This output has shape [128]:
            1. Mean dim=1 -> [1, 128]
            2. Squeeze -> [128]
            Significance: mean of attention scores for each word
            Drawback: scores are too similar
    """
    def __init__(self, config):
        super(BertSelfAttentionScoresP, self).__init__()
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

    def forward(self, hidden_states, attention_mask, head_mask=None, is_visualize_attention=False):
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

        # Visualize attention_probs
        if is_visualize_attention:
            attention_probs_img = attention_probs.squeeze().squeeze().cpu().detach().numpy()
            visualize_attention(attention_probs_img[:30, :30])

        outputs = attention_probs.squeeze().squeeze()
        outputs = torch.mean(outputs, dim=1)
        max_value = torch.max(outputs)
        outputs = torch.div(outputs, max_value)
        if is_visualize_attention:
            attention_probs_img = outputs.unsqueeze(0).cpu().detach().numpy()
            visualize_attention(attention_probs_img[:, :30])

        return outputs


class BertMultiSelfAttentionScoresP(nn.Module):
    """ Gwena's modification

        Based on modeling_bert.py > BertSelfAttention

        Assumes N=1 (num_attention_heads)
        Visualize AttentionScores Matrix

        The original output (attention_probs) has shape [1, N, 128, 128].
        This output has shape [128]:
            1. Mean dim=1 -> [1, 128]
            2. Squeeze -> [128]
            Significance: mean of attention scores for each word
            Drawback: scores are too similar
    """
    def __init__(self, config):
        super(BertMultiSelfAttentionScoresP, self).__init__()
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

    def forward(self, hidden_states, head_mask=None, is_visualize_attention=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        key_layer_transposed = key_layer.transpose(-1, -2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer_transposed)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        outputs = attention_probs.squeeze().squeeze()

        # Visualize attention_probs
        if is_visualize_attention:
            attention_probs_img = outputs.cpu().detach().numpy()
            visualize_attention(attention_probs_img[:30, :30])

        return outputs


class BertSelfAttentionNoAttMask(nn.Module):
    def __init__(self, config):
        super(BertSelfAttentionNoAttMask, self).__init__()
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

    def forward(self, hidden_states, head_mask=None):
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

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config, max_seq_length):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(max_seq_length, max_seq_length)
        self.LayerNorm = BertLayerNorm(max_seq_length, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertMultiAttentionScoresP(nn.Module):
    """ Gwena's modification

        Based on modeling_bert.py > BertAttention
        BertAttention = BertSelfAttention + Normalization + Dropout
    """
    def __init__(self, config, max_seq_length):
        super(BertMultiAttentionScoresP, self).__init__()
        self.self = BertSelfAttentionNoAttMask(config)
        self.output = BertSelfOutput(config, max_seq_length)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, head_mask=None, is_visualize_attention=False):
        self_outputs = self.self(input_tensor, head_mask)
        attention_output = self.output(self_outputs[1], self_outputs[1])  # 1: att matrix
        outputs = attention_output  # (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # Visualize attention_probs
        if is_visualize_attention:
            attention_probs_img = outputs.squeeze().squeeze().cpu().detach().numpy()
            visualize_attention(attention_probs_img[:30, :30])
        return outputs


class BertAttentionClsQuery(nn.Module):
    """ Gwena's modification

        Based on modeling_bert.py > BertAttention
        BertAttention = BertSelfAttention + Normalization + Dropout

        TODO: Visualize AttentionScores Matrix
        Assumes N=1 (num_attention_heads)
        Use CLS token as Query
    """
    def __init__(self, config):
        super(BertAttentionClsQuery, self).__init__()
        self.self = BertSelfAttentionClsQuery(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None, cls_query=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, cls_query=cls_query)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertSelfAttentionClsQuery(nn.Module):
    def __init__(self, config):
        super(BertSelfAttentionClsQuery, self).__init__()
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

    def forward(self, hidden_states, attention_mask, head_mask=None, cls_query=None):
        mixed_query_layer = self.query(cls_query)
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

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
