#!/bin/bash -v

MODEL_TYPE=embraceroberta  # Options=[embraceroberta, embracerobertaconcatatt, embracerobertawithkeyvaluequery, embracerobertawithkeyvaluequeryconcatatt]

MODEL_NAME="${MODEL_TYPE}"
DIM_REDUCTION_METHOD=attention  # Options = [projection, attention]
P_TYPE="attention_clsquery_weights" # Options = [multinomial, attention_clsquery_weights]
MODEL_NAME="${MODEL_NAME}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_NAME}/"
  EVAL_DIR="../../results/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs/${MODEL_NAME}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=7
BS_EVAL=1
for BS_TRAIN in 8; do
  for EPOCH in 100; do
      for DATASET in chatbot; do
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
                  DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                done
              done
          done
      done
  done
done


MODEL_NAME="${MODEL_TYPE}"
DIM_REDUCTION_METHOD=attention  # Options = [projection, attention]
P_TYPE="multinomial" # Options = [multinomial, attention_clsquery_weights]
MODEL_NAME="${MODEL_NAME}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_NAME}/"
  EVAL_DIR="../../results/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs/${MODEL_NAME}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 8; do
  for EPOCH in 100; do
      for DATASET in chatbot; do
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
                  DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                done
              done
          done
      done
  done
done

MODEL_NAME="${MODEL_TYPE}"
DIM_REDUCTION_METHOD=projection  # Options = [projection, attention]
P_TYPE="attention_clsquery_weights" # Options = [multinomial, attention_clsquery_weights]
MODEL_NAME="${MODEL_NAME}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_NAME}/"
  EVAL_DIR="../../results/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs/${MODEL_NAME}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=7
BS_EVAL=1
for BS_TRAIN in 8; do
  for EPOCH in 100; do
      for DATASET in chatbot; do
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
                  DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                done
              done
          done
      done
  done
done


MODEL_NAME="${MODEL_TYPE}"
DIM_REDUCTION_METHOD=projection  # Options = [projection, attention]
P_TYPE="multinomial" # Options = [multinomial, attention_clsquery_weights]
MODEL_NAME="${MODEL_NAME}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_NAME}/"
  EVAL_DIR="../../results/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs/${MODEL_NAME}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 8; do
  for EPOCH in 100; do
      for DATASET in chatbot; do
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
                  DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"
                  EVAL_OUTPUT_FILENAME="eval_results_${TTS}_${STT}"

                  # Eval
                  CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
                done
              done
          done
      done
  done
done
