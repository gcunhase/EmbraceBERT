import torch.nn as nn
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertLayer, BertPooler
from torch.nn import CrossEntropyLoss, MSELoss
from models.bert_utils import BertPreTrainedModel

"""
    Vanilla BERT with option to apply dropout or not
    
    My modification:
        1. Apply dropout: boolean option
        2. Dropout probability in run_classifier.py argparse 
"""


#@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
#    the pooled output) e.g. for GLUE tasks. """,
#    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
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

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, dropout_prob, add_transformer_layer=False):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.add_transformer_layer = add_transformer_layer

        self.bert = BertModel(config)
        if self.add_transformer_layer:
            self.transformer_layer = BertLayer(config)
            self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(dropout_prob)  # config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False, freeze_bert_weights=False):

        # Fine-tune with
        if freeze_bert_weights:
            self.bert.requires_grad = not freeze_bert_weights

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        if self.add_transformer_layer:
            bert_output = outputs[0]
            model_output = self.transformer_layer(bert_output, attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).float(), head_mask=head_mask)
            sequence_output = model_output[0]
            pooled_output = self.pooler(sequence_output)
            outputs = (sequence_output, pooled_output)
        pooled_output = outputs[1]  # CLS

        if apply_dropout:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [bs, 2]
        #logits = self.softmax(logits)

        outputs = (logits,) + outputs[2:] + (pooled_output,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # This criterion combines `log_softmax` and `nll_loss` in a single function
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
