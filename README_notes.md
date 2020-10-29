## Requirements
Tested with Python 3.6.8, PyTorch 1.0.1.post2, CUDA 10.1
```
./pip install --default-timeout=1000 torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install --default-timeout=1000 torch==1.0.1.post2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m spacy download en
source ./anaconda3/etc/profile.d/conda.sh
conda activate my_env
```
> [pytorch-transformers](https://github.com/huggingface/transformers) version from September 6th 2019
> https://pypi.org/project/pytorch-model-summary/

## EmbraceBERT
1. Docking layer **not needed**: modality features all have the same size
2. Embracement layer:
    * Used on the output of BERT to select important features from sequence
    * Output of BERT has shape (batch_size, sequence_length, embedding_size) = `(bs, 128, 768)`
3. Attention layer **added**:
    * Attention is applied to the `[CLS]` token and `embraced token` (both have same shape of `(bs, 768)`), to obtain a single feature vector of same size
    * Obtained feature vector is used as input to a feedforward layer for improved classification 

## Train Model
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
    ```--seed 1 --is_condensed --task_name askubuntu_intent --model_type embraceroberta --model_name_or_path roberta-base --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/askubuntucorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/condensed_embraceroberta_debug/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_embracebert_p_selfattention```

* Add branches (BranchyNet):
    * Add `--add_branches --embracebert_with_branches` to run with branches in hidden BERT-Transformer layers

* `p=softmax(selfattention)`:
    * Train
    ```--seed 1 --p multihead_bertselfattention_in_p --task_name chatbot_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_embracebert_p_selfattention/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_embracebert_p_selfattention```
    ```--seed 1 --p selfattention --task_name chatbot_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_embracebert_p_selfattention/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_embracebert_p_selfattention```
    * Test
    ```--seed 1 --p multihead_bertselfattention_in_p --task_name chatbot_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=1 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/embracebert_p_multihead_bertselfattention/chatbot/complete/chatbot_ep100_bs4_seed1/ --save_best --log_dir ./runs/debug_embracebert_p_selfattention```

* EmbraceBERT_BS: when p is 'multinomial', the chosen features are the same in a batch. In EmbraceBERT_BS, these features are again calculated for each sequence in a batch sequence. So if before `bs=4`, all 4 sequences would have the same feature indexes chosen. In this new model, each one of these 4 sequences has different indexes.
    * Need to run code again to see if the results are better

* EmbraceBERT with Q=BERTc, K,V=BERTi (`run_classifier_bertquery.py`, `models/EmbraceBERTwithQuery.py`)
    * Train
    ```--seed 1 --p multinomial --task_name chatbot_intent --model_type embracebertwithquery --train_bertc --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir_complete data/intent_processed/nlu_eval/chatbotcorpus/ --data_dir data/intent_stterror_data/chatbot/gtts_witai/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --num_train_epochs_bertc 3.0 --output_dir_complete ./results/debug_embracebertwithquery_p_multinomial_bertc/ --output_dir ./results/debug_embracebertwithquery_p_multinomial/ --overwrite_output_dir --overwrite_cache --save_best --log_dir_complete ./runs/debug_embracebertwithquery_p_selfattention_bertc --log_dir ./runs/debug_embracebertwithquery_p_selfattention```

* BERTwithTokens (att/projection):
    ```--seed 1 --task_name chatbot_intent --model_type bertwithprojection --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_bertwithprojection/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_bertwithprojection```    

* BERT with proj(T_[CLS], cat(Embrace, att(T_[CLS], T_all)):
    ```--seed 1 --p multinomial --dimension_reduction_method projection --task_name chatbot_intent --model_type embracebertconcatatt --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_bertwithprojection/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_bertwithprojection```    

* BERTi not initialized
    ```python run_classifier_bertquery.py --seed 1 --dont_initialize_berti --p multinomial --task_name chatbot_intent --model_type embracebertwithquery --train_bertc --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir_complete data/intent_processed/nlu_eval/chatbotcorpus/ --data_dir data/intent_stterror_data/chatbot/gtts_witai/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --learning_rate 2e-5 --num_train_epochs 3.0 --num_train_epochs_bertc 3.0 --output_dir_complete ./results/debug_embracebertwithquery_p_multinomial_bertc/ --output_dir ./results/debug_embracebertwithquery_p_multinomial/ --overwrite_output_dir --overwrite_cache --save_best --log_dir_complete ./runs/debug_embracebertwithquery_p_selfattention_bertc --log_dir ./runs/debug_embracebertwithquery_p_selfattention```

* BERT + 1 Transformer layer
    ```--seed 1 --task_name chatbot_intent --model_type bertplustransformerlayer --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_bertplustransformerlayer/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_bertplustransformerlayer```
    
* RoBERTa
    ```
    --seed 1 --task_name chatbot_intent --model_type robertawithatt --model_name_or_path roberta-base --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_robertawithatt/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_robertawithatt
    --seed 1 --task_name chatbot_intent --model_type embraceroberta --model_name_or_path roberta-base --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_robertawithatt/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_robertawithatt
    --seed 1 --task_name chatbot_intent --model_type robertawithatt --model_name_or_path roberta-base --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_robertawithatt/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_robertawithatt
    --seed 1 --task_name chatbot_intent --model_type embracebert --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_embracebert/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_embracebert
    ```
  

# Calculate number of parameters
```
python run_classifier.py --seed 1 --task_name chatbot_intent --model_type $MODEL_NAME --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# BERT
python run_classifier.py --seed 1 --task_name chatbot_intent --model_type bert --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p multinomial --dimension_reduction_method attention --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+att+p_att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p attention_clsquery_weights --dimension_reduction_method attention --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+proj
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p multinomial --dimension_reduction_method projection --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# EBERT+proj+p_att
--seed 1 --task_name chatbot_intent --model_type embracebertwithkeyvaluequeryconcatatt --p attention_clsquery_weights --dimension_reduction_method projection --model_name_or_path bert-base-uncased --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# RoBERTa
python run_classifier.py --seed 1 --task_name chatbot_intent --model_type roberta --model_name_or_path roberta-base --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
--seed 1 --task_name chatbot_intent --model_type roberta --model_name_or_path roberta-base --logging_steps 1 --do_train --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
# ERoBERTa
--seed 1 --task_name chatbot_intent --model_type embraceroberta --p multinomial --dimension_reduction_method attention --model_name_or_path roberta-base --logging_steps 1 --do_calculate_num_params --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_num_params/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_num_params
```
> MODEL_NAME: 'bert' (109,483,778), 'bertwithatt' (111,253,250), 'bertwithattclsprojection' (111,253,253), 'bertwithprojection' (109,483,908), 'bertwithprojectionatt' (111,253,379),

> $MODEL_NAME2: 'embracebert', 'embracebertconcatatt', 'embracebertwithkeyvaluequery', 'embracebertwithkeyvaluequeryconcatatt'

>               DIM_REDUCTION_METHOD=[attention, projection], P_TYPE=[multinomial, attention_clsquery_weights]
>               EBERT - att (111,253,250), att+p_att (113,022,722), proj (109,483,781), proj+p_att (111,253,253)
>               EBERT_concatatt - att (113,022,722), att+p_att (114,792,194), proj (111,253,254), proj+p_att (113,022,726)
>               EBERTkvq (is_evaluate=True manually in EmbraceBERTwithQuery) - att (113,617,154), att+p_att (115,386,626), proj (111,847,685), proj+p_att (113,617,157)
>               EBERTkvq_concatatt (is_evaluate=True manually in EmbraceBERTwithQuery) - att (115,386,626), att+p_att (117,156,098), proj (113,617,158), proj+p_att (115,386,630)

# Significance testing
```
--seed 1 --task_name chatbot_intent --model_type bertwithatt --model_name_or_path bert-base-uncased --logging_steps 1 --do_lower_case --data_dir data/intent_processed/nlu_eval/chatbotcorpus/ --max_seq_length 128 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./results/debug_robertawithatt/ --overwrite_output_dir --overwrite_cache --save_best --log_dir ./runs/debug_robertawithatt
```
