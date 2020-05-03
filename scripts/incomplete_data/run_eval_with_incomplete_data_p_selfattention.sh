#!/bin/bash -v

# Options: bert, bert_frozen, embracebert, embracebert_frozenbert (and roberta)
MODEL_TYPE=embracebert
P_TYPE="attention_clsquery"
MODEL_NAME="${MODEL_TYPE}_p_${P_TYPE}"
IS_CONDENSED=false  # if true, embracebert_condensed
IS_FROZEN=false  # if true, embracebert_frozenbert
APPLY_DROPOUT=false
DROPOUT_PROB=0.1

if [ "$IS_FROZEN" = true ]; then
  if [[ $MODEL_TYPE = "bert" || $MODEL_TYPE = "roberta" ]]; then
    MODEL_NAME="${MODEL_NAME}_frozen"
  else
    MODEL_NAME="${MODEL_NAME}_frozenbert"
  fi
fi

# shellcheck disable=SC2004
if [ "$IS_CONDENSED" = true ]; then
  MODEL_NAME="${MODEL_NAME}_condensed"
fi

if [ "$APPLY_DROPOUT" = true ]; then
  MODEL_NAME="${MODEL_NAME}_withDropout${DROPOUT_PROB}"
fi

#OUTPUT_DIR="/media/ceslea/DATA/EmbraceBERT-results-backup/models_trained_with_complete_data/${MODEL_NAME}/"
OUTPUT_DIR="../../results/${MODEL_NAME}/"
EVAL_DIR="../../results/test_with_incomplete_results/${MODEL_NAME}/"
mkdir $EVAL_DIR
RUN_DIR="../../runs/${MODEL_NAME}/"

# shellcheck disable=SC2004
if [[ $MODEL_TYPE == *"roberta"* ]]; then
  MODEL_NAME_OR_PATH="roberta-base"
else
  MODEL_NAME_OR_PATH="bert-base-uncased"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4 16; do
  for EPOCH in 100; do
      for DATASET in chatbot; do  # askubuntu webapplications
          echo $DATASET
          echo "Evaluating ${DATASET} dataset with incomplete data for ${EPOCH} epochs"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              EVAL_PATH_1="${EVAL_DIR}/${DATASET}"
              mkdir $EVAL_PATH_1
              EVAL_PATH="${EVAL_DIR}/${RESULT_DIR}"
              mkdir $EVAL_PATH

              for TTS in "gtts" "macsay"; do
                for STT in "google" "sphinx" "witai"; do
                  DATA_DIR="../../data/intent_stterror_data/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  if [ "$IS_CONDENSED" = true ]; then
                    if [ "$IS_FROZEN" = true ]; then
                      if [ "$APPLY_DROPOUT" = true ]; then
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --is_condensed --freeze_bert_weights --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      else
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --is_condensed --freeze_bert_weights --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      fi
                    else
                      if [ "$APPLY_DROPOUT" = true ]; then
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --is_condensed --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      else
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --is_condensed --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      fi
                    fi
                  else
                    if [ "$IS_FROZEN" = true ]; then
                      if [ "$APPLY_DROPOUT" = true ]; then
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --freeze_bert_weights --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      else
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --freeze_bert_weights --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      fi
                    else
                      if [ "$APPLY_DROPOUT" = true ]; then
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      else
                        CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --share_branch_weights --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                      fi
                    fi
                  fi
                done
              done
          done
      done
  done
done
