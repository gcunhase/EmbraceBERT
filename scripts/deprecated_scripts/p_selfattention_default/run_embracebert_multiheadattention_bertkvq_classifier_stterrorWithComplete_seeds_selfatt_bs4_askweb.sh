#!/bin/bash -v

MODEL_TYPE=embracebertwithkeyvaluequery
P_TYPE="selfattention"  #"selfattention"
LR=2e-5
#OUTPUT_DIR="/media/ceslea/DATA/EmbraceBERT-results-backup/${MODEL_TYPE}_p_${P_TYPE}/"
OUTPUT_DIR="../../results/${MODEL_TYPE}_p_${P_TYPE}/"
RUN_DIR="../../runs/${MODEL_TYPE}_p_${P_TYPE}/"

BS_EVAL=1
for BS_TRAIN in 4; do # 16; do  #4 16; do
  for DATASET in askubuntu webapplications; do
      echo $DATASET
      for EPOCH in 100; do # 100; do
          echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

          for STT in gtts macsay; do
            for TTS in google sphinx witai; do
              STT_TTS="${STT}_${TTS}"
              DATA_DIR="../../data/intent_stterror_data_withComplete/${DATASET}/${STT_TTS}/"

              for SEED in 1 2 3 4 5 6 7 8 9 10; do
                  RESULT_DIR="${DATASET}/stterror_withComplete/${STT_TTS}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
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
  done
done