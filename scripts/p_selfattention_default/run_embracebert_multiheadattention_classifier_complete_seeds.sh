#!/bin/bash -v

MODEL_TYPE=embracebert
P_TYPE="attention_clsquery_weights"  #"multiple_multihead_bertattention_in_p"  #"attention_clsquery"  #"multiple_multihead_bertselfattention_in_p"  #"multihead_bertselfattention_in_p" #"multihead_bertattention"
LR=2e-5
OUTPUT_DIR="../../results/${MODEL_TYPE}_p_${P_TYPE}/"
RUN_DIR="../../runs/${MODEL_TYPE}_p_${P_TYPE}/"

BS_EVAL=1
for BS_TRAIN in 4 16; do  #4 16; do
  for DATASET in chatbot; do  # webapplications chatbot; do
      echo $DATASET
      for EPOCH in 100; do # 100; do
          echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

          DATA_DIR="../../data/intent_processed/nlu_eval/${DATASET}corpus/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="${DATASET}/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              # --evaluate_during_training only to be used when training on 1 GPU
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate $LR --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate $LR --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
  done
done
