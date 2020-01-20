#!/bin/bash -v

# Options: bert, bert_frozen, embracebert, embracebert_frozenbert (and roberta)
MODEL_TYPE=bert
IS_CONDENSED=false  # if true, embracebert_condensed

# shellcheck disable=SC2004
if [ "$IS_CONDENSED" = true ]; then
  MODEL_NAME="${MODEL_TYPE}_condensed"
else
  MODEL_NAME="${MODEL_TYPE}"
#  if [ $MODEL_TYPE == "bert" ]; then
#    MODEL_NAME="${MODEL_TYPE}_frozen"
#  else
#    if [ $MODEL_TYPE == "roberta" ]; then
#      MODEL_NAME="${MODEL_TYPE}_frozen"
#    else
#      MODEL_NAME="${MODEL_TYPE}_frozenbert"
#    fi
#  fi
fi

OUTPUT_DIR="/media/ceslea/DATA/EmbraceBERT-results-backup/models_trained_with_complete_data/${MODEL_NAME}/"
EVAL_DIR="../results/test_with_incomplete_results/${MODEL_NAME}/"
mkdir $EVAL_DIR
RUN_DIR="../runs/${MODEL_NAME}/"

# shellcheck disable=SC2004
if [[ $MODEL_TYPE == *"bert"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
else
  MODEL_NAME_OR_PATH="roberta-base"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4 16; do
  for EPOCH in 30 100; do
      for DATASET in askubuntu chatbot webapplications; do # chatbot webapplications
          echo $DATASET
          echo "Evaluating ${DATASET} dataset with incomplete data for ${EPOCH} epochs"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              EVAL_PATH_1="${EVAL_DIR}/${DATASET}"
              mkdir $EVAL_PATH_1
              EVAL_PATH="${EVAL_DIR}/${RESULT_DIR}"
              mkdir $EVAL_PATH

              for TTS in "gtts" "macsay"; do
                for STT in "google" "sphinx" "witai"; do
                  DATA_DIR="../data/intent_stterror_data/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  if [ "$IS_CONDENSED" = true ]; then
                    CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --is_condensed --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                  else
                    CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                  fi
                done
              done
          done
      done
  done
done


BS_EVAL=1
for BS_TRAIN in 16 32; do
  for EPOCH in 3; do
      for DATASET in snips; do # chatbot webapplications
          echo $DATASET
          echo "Evaluating ${DATASET} dataset with incomplete data for ${EPOCH} epochs"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              EVAL_PATH_1="${EVAL_DIR}/${DATASET}"
              mkdir $EVAL_PATH_1
              EVAL_PATH="${EVAL_DIR}/${RESULT_DIR}"
              mkdir $EVAL_PATH

              for TTS in "gtts" "macsay"; do
                for STT in "google" "sphinx" "witai"; do
                  DATA_DIR="../data/intent_stterror_data/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  if [ "$IS_CONDENSED" = true ]; then
                    CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --is_condensed --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                  else
                    CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --freeze_bert_weights --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                  fi
                done
              done
          done
      done
  done
done
