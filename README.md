## About
* *EmbraceBERT*: BERT with ideas from [EmbraceNet](https://arxiv.org/abs/1904.09078) to improve robustness and thus classification accuracy in noisy data.
* 3 settings:
    1. Trained and tested with complete data
    2. Trained with complete data and tested with incomplete data
    3. Trained and tested with incomplete data

## Contents
[Requirements](#requirements) • [EmbraceBERT](#embracebert) • [How to Use](#how-to-use) • [Results](#results) • [How to Cite](#acknowledgement)

## Requirements
Tested with Python 3.6.8, PyTorch 1.0.1.post2, CUDA 10.1
```
pip install --default-timeout=1000 torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m spacy download en
source ./anaconda3/etc/profile.d/conda.sh
conda activate my_env
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
* Proposed: all tokens (BERT, EBERT, EBERTkvq)
    ```
    # BERT with tokens
    ./scripts/[DIR_SETTING_1_OR_3]/run_bertWithTokens_classifier_seeds.sh
    # EBERT
    ./scripts/[DIR_SETTING_1_OR_3]/run_embracebert_classifier_seeds.sh
    # EBERTkvq
    ./scripts/[DIR_SETTING_1_OR_3]/run_embracebert_multiheadattention_bertkvq_classifier_seeds.sh
    ```

* Baseline (BERT)
    ```
    ./scripts/[DIR_SETTING_1_OR_3]/run_bert_classifier_seeds.sh
    ```
    
### 3. Test model with Incomplete data
```
./scripts/[DIR_SETTING_2]/run_eval_with_incomplete_data.sh
```
> Modify script with the path and type of your model 

### 4. Calculate number of parameters
```
python run_classifier.py --seed 1 --task_name chatbot_intent --model_type $MODEL_NAME --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
python run_classifier.py --seed 1 --task_name chatbot_intent --model_type bert --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p multinomial --dimension_reduction_method attention --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+att+p_att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p attention_clsquery_weights --dimension_reduction_method attention --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+proj
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p multinomial --dimension_reduction_method projection --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+proj+p_att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p attention_clsquery_weights --dimension_reduction_method projection --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
```
> MODEL_NAME: 'bert' (109,483,778), 'bertwithatt' (111,253,250), 'bertwithattclsprojection' (111,253,253), 'bertwithprojection' (109,483,908), 'bertwithprojectionatt' (111,253,379),

> $MODEL_NAME2: 'embracebert', 'embracebertconcatatt', 'embracebertwithkeyvaluequery', 'embracebertwithkeyvaluequeryconcatatt'

>               DIM_REDUCTION_METHOD=[attention, projection], P_TYPE=[multinomial, attention_clsquery_weights]
>               EBERT - att (111,253,250), att+p_att (113,022,722), proj (109,483,781), proj+p_att (111,253,253)
>               EBERT_concatatt - att (113,022,722), att+p_att (114,792,194), proj (111,253,254), proj+p_att (113,022,726)
>               EBERTkvq (is_evaluate=True manually in EmbraceBERTwithQuery) - att (113,617,154), att+p_att (115,386,626), proj (111,847,685), proj+p_att (113,617,157)
>               EBERTkvq_concatatt (is_evaluate=True manually in EmbraceBERTwithQuery) - att (115,386,626), att+p_att (117,156,098), proj (113,617,158), proj+p_att (115,386,630)

### Output    
| File                              | Description |
| --------------------------------- | ----------- |
| `checkpoint-best-${EPOCH_NUMBER}` | Directory with saved model |
| `eval_results.json`               | JSONified train/eval information |
| `eval_results.txt`                | Train/eval information: eval accuracy and loss, global_step and train loss |

## Results
### Baseline
[BERT/RoBERTa](https://github.com/gcunhase/IntentClassifier-RoBERTa) and [NLU Services](https://github.com/gcunhase/IntentClassifier) [[more info](https://github.com/gcunhase/IntentClassifier-RoBERTa)]

### F1-scores (English)
[AskUbuntu](./results_notes/askubuntu.md) • [Chatbot](./results_notes/chatbot.md) • [WebApplications](./results_notes/webapplications.md) • [Snips](./results_notes/snips.md)

### F1-scores (Korean)
[Chatbot](./results_notes/chatbot_korean.md)

## Acknowledgement
In case you wish to use this code, please credit this repository or send me an email at `gwena.cs@gmail.com` with any requests or questions.

Code based on [HuggingFace's repository](https://github.com/huggingface/transformers).
