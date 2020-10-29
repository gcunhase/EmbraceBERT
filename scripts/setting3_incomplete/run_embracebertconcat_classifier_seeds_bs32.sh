#!/bin/bash -v

MODEL_TYPE=embracebertconcatatt  # Options = [embracebert, embracebertconcatatt]
DIM_REDUCTION_METHOD=projection  # Options = [projection, attention]
P_TYPE="attention_clsquery_weights"  # Options = [multinomial, attention_clsquery_weights]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"
  DATA_PATH_NAME="intent_stterror_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"
  DATA_PATH_NAME="korean_intent_stterror_data"
fi
echo $MODEL_NAME_OR_PATH

BS_TRAIN=32 # 8
BS_EVAL=1
for DATASET in snips; do # chatbot; do
    echo $DATASET
      for TTS in "macsay"; do # "gtts" "macsay"; do
          for STT in "sphinx"; do # "google" "sphinx" "witai"; do
            for EPOCH in 100; do
                echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs"

                DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/${TTS}_${STT}/"

                for SEED in 1 2 3 4 5 6 7 8; do
                    RESULT_DIR="${DATASET}/stterror/${TTS}_${STT}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
                    OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
                    LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

                    # Train
                    CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                    # Eval
                    CUDA_VISIBLE_DEVICES=5 python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                done
            done
        done
    done
done
