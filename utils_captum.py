from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import matplotlib.pyplot as plt


"""
Source: https://colab.research.google.com/drive/1Lw3JTZio03VwPvSVFzLJmZ52oBRpo9ZM
        https://gist.github.com/davidefiocco/47137f6eb7e3351c9bac4580c2ccc9d4
"""


# def predict(inputs, model):
#     return model(inputs)[0]


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, tokenizer, device):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0, device=torch.device("cpu")):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids, device):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_class",
        "attr_score",
        "raw_input",
        "convergence_score",
        "model_type",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input,
        convergence_score,
        model_type,
    ):
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input = raw_input
        self.convergence_score = convergence_score
        self.model_type = model_type


from captum.attr import LayerIntegratedGradients
def visualize_model_single_output(args, tokenizer, model, text, true_label, device, custom_forward, predict):
    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence
    lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id,
                                                                tokenizer, device)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id, device)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids, device)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    # Check predict output
    pred_score = predict(input_ids)
    if not args.model_type == 'bert':
        pred_score = pred_score.unsqueeze(0)

    # Check output of custom_forward
    custom_forward(input_ids)  # torch.cat([input_ids])

    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        n_steps=7000,  # 700
                                        internal_batch_size=5,  # 3
                                        return_convergence_delta=True)

    pred_label = torch.argmax(pred_score[0]).detach().cpu().item()  # .cpu().numpy()
    pred_prob = torch.softmax(pred_score, dim=1)[0][pred_label].detach().cpu().item()  # .cpu().detach().numpy()
    print('Sentence: ', text)
    print('Sentiment: ' + str(pred_label) + \
          ', Probability positive: ' + str(pred_prob))

    attributions_sum = summarize_attributions(attributions)
    # storing couple samples in an array for visualization purposes
    score_vis = VisualizationDataRecord(attributions_sum,
                                        pred_prob,  # torch.softmax(score, dim=1)[0][1], # [0][0]
                                        pred_label,  # torch.argmax(torch.softmax(score, dim=1)[0]),  # dim = 0)[0])
                                        true_label,  # 1  # true class
                                        text,
                                        attributions_sum.sum(),
                                        all_tokens,
                                        delta,
                                        args.model_type)
    return score_vis


from captum.attr._utils.visualization import format_classname, format_word_importances, _get_color
from typing import Any, Iterable, List, Tuple, Union
try:
    from IPython.core.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

def visualize_text_custom(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = True, save_img: bool = True, html_filename: str = 'temp.html'
) -> None:
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Model</th>"
        "<th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
        "<tr><td><hr></td></tr>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.model_type),
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{0} ({1:.4f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )
        if datarecord.model_type == 'bert':
            rows.append('<tr><td><hr></td></tr>')

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 1], ["DepartureTime (0)", "FindConnection (1)"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html_obj = HTML("".join(dom))
    display(html_obj)
    if save_img:
        import os
        import webbrowser

        path = os.path.abspath(html_filename)  # 'temp.html')
        url = 'file://' + path

        with open(path, 'w') as f:
            f.write(html_obj.data)
        webbrowser.open(url)
        print("Save img")
