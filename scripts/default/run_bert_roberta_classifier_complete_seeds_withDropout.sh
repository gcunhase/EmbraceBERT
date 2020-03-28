#!/bin/bash -v

MODEL_TYPE=bert
OUTPUT_DIR_ROOT="../../results/${MODEL_TYPE}_withDropout"
RUN_DIR_ROOT="../../runs/${MODEL_TYPE}_withDropout"

if [ $MODEL_TYPE == "bert" ]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
else
  MODEL_NAME_OR_PATH="roberta-base"
fi

BS_EVAL=1
for BS_TRAIN in 16 32; do
  for DATASET in snips; do
    for DROPOUT_PROB in 0.1 0.3; do
      OUTPUT_DIR="${OUTPUT_DIR_ROOT}${DROPOUT_PROB}/"
      RUN_DIR="${RUN_DIR_ROOT}${DROPOUT_PROB}/"
      echo $DATASET
      for EPOCH in 3; do  # 30 100; do  # 30 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          # DATA_DIR="../../data/intent_processed/nlu_eval/${DATASET}corpus/"
          DATA_DIR="../../data/intent_processed/${DATASET}/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
    done
  done
done

MODEL_TYPE=roberta
OUTPUT_DIR_ROOT="../../results/${MODEL_TYPE}_withDropout"
RUN_DIR_ROOT="../../runs/${MODEL_TYPE}_withDropout"

if [ $MODEL_TYPE == "bert" ]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
else
  MODEL_NAME_OR_PATH="roberta-base"
fi

BS_EVAL=1
for BS_TRAIN in 16 32; do
  for DATASET in snips; do
    for DROPOUT_PROB in 0.1 0.3; do
      OUTPUT_DIR="${OUTPUT_DIR_ROOT}${DROPOUT_PROB}/"
      RUN_DIR="${RUN_DIR_ROOT}${DROPOUT_PROB}/"
      echo $DATASET
      for EPOCH in 3; do  # 30 100; do  # 30 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          # DATA_DIR="../../data/intent_processed/nlu_eval/${DATASET}corpus/"
          DATA_DIR="../../data/intent_processed/${DATASET}/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --apply_dropout --dropout_prob $DROPOUT_PROB --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
    done
  done
done

