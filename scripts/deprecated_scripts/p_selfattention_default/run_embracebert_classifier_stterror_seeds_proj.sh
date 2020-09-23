#!/bin/bash -v

MODEL_TYPE=embracebert
DIM_REDUCTION_METHOD=projection
P_TYPE="attention_clsquery_weights"
OUTPUT_DIR="../../results/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"
RUN_DIR="../../runs/${MODEL_TYPE}_${DIM_REDUCTION_METHOD}_p_${P_TYPE}/"

BS_TRAIN=8
BS_EVAL=1
for DATASET in chatbot; do
    echo $DATASET
    for TTS in "gtts" "macsay"; do
        for STT in "google" "sphinx" "witai"; do
            for EPOCH in 100; do  # 30 100; do
                echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs"

                DATA_DIR="../../data/intent_stterror_data/${DATASET}/${TTS}_${STT}/"

                for SEED in 1 2 3 4 5 6 7 8 9 10; do
                    RESULT_DIR="${DATASET}/stterror/${TTS}_${STT}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
                    OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
                    LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

                    # Train
                    CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                    # Eval
                    CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --p $P_TYPE --dimension_reduction_method $DIM_REDUCTION_METHOD --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
                done
            done
        done
    done
done