import torch
import torch.nn as nn
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertEmbeddings
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from torchnlp.nn.attention import Attention
# torchnlp.Attention: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# class BertForSequenceClassification(BertPreTrainedModel):
class EmbraceBertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForSequenceClassification(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(EmbraceBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # 768
        # self.args = args

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embrace_attention = Attention(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # Freeze BERT's weights
        #if freeze_berts_weights:
        #    self.bert.requires_grad = not freeze_berts_weights

        self.apply(self.init_weights)
        # self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False):
        # encoder_output = bs x 128 x 768
        bert_output = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

        # pooled_enc_output = bs x 768
        output_tokens_from_bert = bert_output[0]
        cls_output = bert_output[1]  # CLS
        # pooled_output = self.dropout(cls_output)

        # Docking layer not needed given that all features have the same size
        # Embracement layer with all outputs (except CLS)
        tokens_to_embrace = output_tokens_from_bert[:, 1:, :]  # (8, 128, 768) = (bs, sequence_length (where the first index is CLS), embedding_size)
        [bs, seq_len, emb_size] = tokens_to_embrace.size()
        tokens_to_embrace = tokens_to_embrace.cpu().detach().numpy()
        # Consider each token in the sequence of 128 as one modality.
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

        # Apply attention layer to CLS and embraced_features_token
        query = torch.unsqueeze(cls_output, 1).cuda()  # query = torch.randn(5, 1, 256)
        context = torch.unsqueeze(embraced_features_token, 1).cuda()  # context = torch.randn(5, 5, 256)
        embrace_output, weights = self.embrace_attention(query, context)
        embrace_output = embrace_output.squeeze()

        # No need because the embrace layer function as a dropout mechanism?
        if apply_dropout:
            embrace_output = self.dropout(embrace_output)

        # Classify
        logits = self.classifier(embrace_output)

        # a_bert_output = bert_output[2:]
        # a_embrace_output = embrace_output
        outputs = (logits,) + bert_output[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # loss and probability of each class (vector)

        return outputs  # (loss), logits, (hidden_states), (attentions)