#!/bin/bash -v

MODEL_TYPE=embracebertwithqueryconcatatt  #embracebertwithquery
DIM_REDUCTION_METHOD=projection
P_TYPE="attention_clsquery_weights"
LR=2e-5
OUTPUT_DIR_COMPLETE="../../results/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}_noInitBerti_bertc/"
RUN_DIR_COMPLETE="../../runs/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}_noInitBerti_bertc/"
OUTPUT_DIR="../../results/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}_noInitBerti/"
RUN_DIR="../../runs/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}_noInitBerti/"

BS_EVAL=1
for BS_TRAIN in 8; do  # 16; do  #4 16; do
  for DATASET in chatbot; do  # webapplications chatbot; do
      echo $DATASET
      EPOCH_BERTC=10
      for EPOCH in 100; do # 100; do
          echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

          DATA_DIR_COMPLETE="../../data/intent_processed/nlu_eval/${DATASET}corpus/"
          for STT in gtts macsay; do
            for TTS in google sphinx witai; do
              STT_TTS="${STT}_${TTS}"
              DATA_DIR="../../data/intent_stterror_data/${DATASET}/${STT_TTS}/"

              for SEED in 1 2 3 4 5 6 7 8 9 10; do
                  RESULT_DIR="${DATASET}/stterror/${STT_TTS}/${DATASET}_ep${EPOCH}_epQ${EPOCH_BERTC}_bs${BS_TRAIN}_seed${SEED}"
                  #RESULT_DIR="${DATASET}/stterror/${STT_TTS}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
                  OUT_PATH_COMPLETE="${OUTPUT_DIR_COMPLETE}/${RESULT_DIR}"
                  LOG_DIR_PATH_COMPLETE="${RUN_DIR_COMPLETE}/${RESULT_DIR}"
                  OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
                  LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

                  # Train
                  # --evaluate_during_training only to be used when training on 1 GPU
                  CUDA_VISIBLE_DEVICES=2 python ../../run_classifier_bertquery.py --seed $SEED --dont_initialize_berti --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --train_bertc --evaluate_during_training --do_lower_case --data_dir_complete $DATA_DIR_COMPLETE --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate $LR --num_train_epochs $EPOCH --num_train_epochs_bertc $EPOCH_BERTC --output_dir_complete $OUT_PATH_COMPLETE --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir_complete $LOG_DIR_PATH_COMPLETE --log_dir $LOG_DIR_PATH
                  # Eval
                  CUDA_VISIBLE_DEVICES=2 python ../../run_classifier_bertquery.py --seed $SEED --dont_initialize_berti --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_eval --do_lower_case --data_dir_complete $DATA_DIR_COMPLETE --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate $LR --num_train_epochs $EPOCH --num_train_epochs_bertc $EPOCH_BERTC --output_dir_complete $OUT_PATH_COMPLETE --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir_complete $LOG_DIR_PATH_COMPLETE --log_dir $LOG_DIR_PATH
              done
            done
          done
      done
  done
done
