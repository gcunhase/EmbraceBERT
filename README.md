## About
*EmbraceBERT*: BERT with ideas from [EmbraceNet](https://arxiv.org/abs/1904.09078) to improve robustness and thus classification accuracy in noisy data.

## Contents
[Requirements](#requirements) • [EmbraceBERT](#embracebert) • [How to Use](#how-to-use) • [Results](#results) • [How to Cite](#acknowledgement)

## Requirements
Tested with Python 3.6.8, PyTorch 1.0.1.post2, CUDA 10.1
```
pip install -r requirements.txt
python -m spacy download en
```
> [pytorch-transformers](https://github.com/huggingface/transformers) version from September 6th 2019

## EmbraceBERT
1. Docking layer **not needed**: modality features all have the same size
2. Embracement layer:
    * Used on the output of BERT to select important features from sequence
    * Output of BERT has shape (batch_size, sequence_length, embedding_size) = `(bs, 128, 768)`
3. Attention layer **added**:
    * Attention is applied to the `[CLS]` token and `embraced token` (both have same shape of `(bs, 768)`), to obtain a single feature vector of same size
    * Obtained feature vector is used as input to a feedforward layer for improved classification 

## How to Use
### 1. Dataset
* Open-source NLU benchmarks (SNIPS, Chatbot, Ask Ubuntu and Web Applications Corpora)
* Available in `data` directory [[more info](https://github.com/gcunhase/IntentClassifier-RoBERTa/data/README.md)] 

### 2. Train Model
* Debug
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_classifier.py --seed 1 --task_name askubuntu_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/embracebert_debug_ep3_bs4_seed1/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/embracebert_debug_ep3_bs4_seed1
    CUDA_VISIBLE_DEVICES=0 python run_classifier.py --seed 1 --task_name askubuntu_intent --model_type embraceroberta --model_name_or_path roberta-base --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/embraceroberta_debug/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/embraceroberta_debug
    CUDA_VISIBLE_DEVICES=0 python run_classifier.py --seed 1 --is_condensed --task_name askubuntu_intent --model_type embraceroberta --model_name_or_path roberta-base --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/condensed_embraceroberta_debug/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/condensed_embraceroberta_debug
    ```
* EmbraceBERT fine-tuned with Intention Classification Dataset
    * REDO EXPERIMENTS WITH JUST EmbraceBERT
    ```
    # Wrong:
    tokens_to_embrace = output_tokens_from_bert[:, 1:, :]  # (8, 128, 768) = (bs, sequence_length (where the first index is CLS), embedding_size)
    # Corrected to:
    tokens_to_embrace = output_tokens_from_bert[:, :, :]  # (8, 128, 768) = (bs, sequence_length, embedding_size)
    ```
    > This is correct in CondensedEmbraceBERT

    ```
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_embracebert_classifier_askubuntu_complete_seeds.sh
    ```
    > For EmbraceRoBERTa, change `--model_type` to `embraceroberta` and `--model_name_or_path` to `roberta-base`

    > `CUDA_VISIBLE_DEVICES=0 python run_embracebert_classifier.py --seed 1 --task_name askubuntu_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/embracebert/askubuntu_complete_ep3_bs4_seed1/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/embracebert/askubuntu_complete_ep3_bs4_seed1`

* Condensed EmbraceBERT fine-tuned with Intention Classification Dataset: add `--is_condensed` to run
    * In this version, the embracement layer doesn't consider all the tokens, only those referring to tokens in the original sentence. Previously, if the max length of the sequence was 128, all 128 tokens would be considered even if the sentence only had 10 tokens in it.
    
* Frozen EBERT: freeze BERT weights and to end-to-end finetuning after embrace layer with classifier loss is saturated for a few steps
    * Add `--freeze_bert_weights --num_train_epochs_frozen_bert 100.0` to run
    ```--seed 1 --is_condensed --task_name askubuntu_intent --model_type embraceroberta --model_name_or_path roberta-base --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/condensed_embraceroberta_debug/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/condensed_embraceroberta_debug```

* Add branches (BranchyNet):
    * Add `--add_branches --embracebert_with_branches` to run with branches in hidden BERT-Transformer layers

### 3. Test model with Incomplete data
```
./run_eval_with_incomplete_data.sh
```

### Output    
| File | Description |
| ---- | ----------- |
| `checkpoint-best-${EPOCH_NUMBER}` | Directory with saved model |
| `eval_results.json` | JSONified train/eval information |
| `eval_results.txt` | Train/eval information: eval accuracy and loss, global_step and train loss |

## Results
### Baseline
[BERT/RoBERTa](https://github.com/gcunhase/IntentClassifier-RoBERTa) and [NLU Services](https://github.com/gcunhase/IntentClassifier) [[more info](https://github.com/gcunhase/IntentClassifier-RoBERTa)]

### F1-scores
[AskUbuntu](./results_notes/askubuntu.md) • [Chatbot](./results_notes/chatbot.md) • [WebApplications](./results_notes/webapplications.md) • [Snips](./results_notes/snips.md)

## Acknowledgement
In case you wish to use this code, please credit this repository or send me an email at `gwena.cs@gmail.com` with any requests or questions.

Code based on [HuggingFace's repository](https://github.com/huggingface/transformers).
