#!/bin/bash -v

MODEL_TYPE=robertawithprojection  # Options=[robertawithatt, robertawithprojection, robertawithprojectionatt, robertawithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 8; do
  for DATASET in chatbot; do
      echo $DATASET
      for EPOCH in 100; do
      for TTS in "gtts" "macsay"; do
          for STT in "google" "sphinx" "witai"; do
              echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs and bs ${BS_TRAIN}"
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"

              for SEED in 1 2 3 4 5 6 7 8 9 10; do
                  RESULT_DIR="${DATASET}/stterror/${TTS}_${STT}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
                  OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
                  LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

                  # Train
                  CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                  # Eval
                  CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              done
            done
          done
      done
  done
done

MODEL_TYPE=robertawithprojectionatt  # Options=[robertawithatt, robertawithprojection, robertawithprojectionatt, robertawithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-uncased"
  fi
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  if [[ $MODEL_TYPE == *"roberta"* ]]; then
    MODEL_NAME_OR_PATH="xlm-roberta-base"
  else
    MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  fi
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 8; do
  for DATASET in chatbot; do
      echo $DATASET
      for EPOCH in 100; do
      for TTS in "gtts" "macsay"; do
          for STT in "google" "sphinx" "witai"; do
              echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs and bs ${BS_TRAIN}"
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"

              for SEED in 1 2 3 4 5 6 7 8 9 10; do
                  RESULT_DIR="${DATASET}/stterror/${TTS}_${STT}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
                  OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
                  LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

                  # Train
                  CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                  # Eval
                  CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              done
            done
          done
      done
  done
done
