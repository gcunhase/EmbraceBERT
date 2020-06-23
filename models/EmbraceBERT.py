import torch
import torch.nn as nn
# from pytorch_transformers import *
from pytorch_transformers import BertModel, modeling_bert

from torch.nn import CrossEntropyLoss, MSELoss
from models.EmbracementLayer import EmbracementLayer
from models.CondensedEmbracementLayer import CondensedEmbracementLayer
from models.bert_utils import BertPreTrainedModel
from models.AttentionLayer import AttentionLayer
from models.BranchesLayer import BranchesLayer

import numpy as np


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
    # EmbraceBERT with branches: added 'add_branches' and 'share_branch_weights'
    def __init__(self, config, dropout_prob, is_condensed=False, add_branches=False,
                 share_branch_weights=False, p='multinomial', max_seq_length=128,
                 dimension_reduction_method='attention', concat_att_with_embracement=False):
        super(EmbraceBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # 768
        self.is_condensed = is_condensed
        self.p = p
        self.max_seq_length = max_seq_length  # 128
        self.dimension_reduction_method = dimension_reduction_method
        self.concat_att_with_embracement = concat_att_with_embracement

        """EmbraceBERT with branches"""
        self.num_labels_evaluator = 2
        self.add_branches = add_branches
        self.share_branch_weights = share_branch_weights
        """END MODIFICATION"""

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)  # config.hidden_dropout_prob)
        if not self.is_condensed:
            self.embracement_layer = EmbracementLayer(config, self.hidden_size, self.p, self.max_seq_length)
        else:
            #self.embracement_layer = CondensedEmbracementLayer(config, self.hidden_size, self.p)
            self.embracement_layer = CondensedEmbracementLayer(self.hidden_size, self.p)

        if self.dimension_reduction_method == 'attention':
            self.attention = AttentionLayer(self.hidden_size)
        else:  # projection
            self.projection = nn.Linear(2, 1)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        if self.concat_att_with_embracement:  # BERT with proj(T_[CLS], cat(Embrace, att(T_[CLS], T_all)):
            # Att cls layer
            self.embrace_attention = AttentionLayer(self.hidden_size)
            # Projection between 3 inputs: T_[CLS], T_e, T_att
            self.projection = nn.Linear(3, 1)

        """EmbraceBERT with branches"""
        if self.add_branches:
            self.branches_layer = BranchesLayer(config, share_branch_weights, num_labels_evaluator=self.num_labels_evaluator)
        """END MODIFICATION"""

        # Freeze BERT's weights
        #if freeze_berts_weights:
        #    self.bert.requires_grad = not freeze_berts_weights

        self.apply(self.init_weights)
        # self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False, freeze_bert_weights=False):

        # Fine-tune with
        if freeze_bert_weights:
            # self.bert.requires_grad = not freeze_bert_weights
            self.bert.training = not freeze_bert_weights

        # encoder_output = bs x 128 x 768
        bert_output = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

        # pooled_enc_output = bs x 768
        # output_tokens_from_bert = bert_output[0]
        cls_output = bert_output[1]  # CLS
        output_tokens_from_bert = bert_output[0]

        """EmbraceBERT with branches"""
        if self.add_branches:
            hidden_tokens_from_bert = bert_output[2]
            output_tokens_from_bert, attention_mask, logits_branches, logits_branches_evaluator, labels_branch_evaluator = self.branches_layer(hidden_tokens_from_bert, output_tokens_from_bert, attention_mask, labels)
        """END MODIFICATION"""

        if self.is_condensed:  # Embracement layer with outputs between CLS and SEP only
            embraced_features_token = self.embracement_layer(output_tokens_from_bert, attention_mask)
        else:  # Embracement layer with all outputs (except CLS)
            # cls_token only used for 'multihead_bertattention_clsquery'
            embraced_features_token = self.embracement_layer(output_tokens_from_bert, cls_token=cls_output)

        if self.concat_att_with_embracement:  # BERT with proj(T_[CLS], cat(Embrace, att(T_[CLS], T_all)):
            # Att cls layer: 'bertwithattclsprojection'
            att_output = self.embrace_attention(cls_output, output_tokens_from_bert)
            att_output = att_output[0]
            if len(att_output.shape) == 1:
                att_output = att_output.unsqueeze(0).unsqueeze(0)
            elif len(att_output.shape) == 2:
                att_output = att_output.unsqueeze(1)
            if self.dimension_reduction_method == 'attention':
                tokens = torch.cat((att_output, embraced_features_token.unsqueeze(1).cuda()), 1)
                embrace_output = self.attention(cls_output.unsqueeze(1), tokens)
                embrace_output = embrace_output[0]
            else:
                tokens = torch.cat((cls_output.unsqueeze(1), att_output, embraced_features_token.unsqueeze(1).cuda()), 1)
                tokens = tokens.permute((0, 2, 1))
                # Projection layer to obtain 1 feature vector for classification
                embrace_output = self.projection(tokens).squeeze()
        else:
            # Last step: Apply attention layer to CLS and embraced_features_token
            # embrace_output = self.embrace_attention(embraced_cls_with_branches, embraced_features_token)
            if self.dimension_reduction_method == 'attention':
                embrace_output = self.attention(cls_output, embraced_features_token)
                embrace_output = embrace_output[0]
            else:  # projection
                tokens = torch.cat((cls_output.unsqueeze(1), embraced_features_token.cuda().unsqueeze(1)), 1)
                tokens = tokens.permute((0, 2, 1))
                # Projection layer to obtain 1 feature vector for classification
                embrace_output = self.projection(tokens).squeeze()

        # No need because the embrace layer functions as a dropout mechanism?
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

            """EmbraceBERT with branches: Add losses from all branches"""
            if self.add_branches:
                loss_branches, loss_branches_evaluator = \
                    self.branches_layer.loss_branches_and_evaluator(loss_fct, logits_branches,
                                                                    logits_branches_evaluator, labels,
                                                                    labels_branch_evaluator)
                loss += loss_branches + loss_branches_evaluator
            """END MODIFICATION"""

            outputs = (loss,) + outputs  # loss and probability of each class (vector)

        return outputs  # (loss), logits, (hidden_states), (attentions)
