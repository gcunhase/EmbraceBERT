#!/bin/bash -v

MODEL_TYPE=embracebert
OUTPUT_DIR="../results/${MODEL_TYPE}_embraceLayerAsModule/"
RUN_DIR="../runs/${MODEL_TYPE}_embraceLayerAsModule/"

BS_TRAIN=16
BS_EVAL=1
for DATASET in askubuntu; do  # chatbot askubuntu; do
    echo $DATASET
    for EPOCH in 3 30 100; do  # 30 100; do
        echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

        DATA_DIR="../data/intent_processed/nlu_eval/${DATASET}corpus/"

        for SEED in 1 2 3 4 5 6 7 8 9 10; do
            RESULT_DIR="${DATASET}/${DATASET}_complete_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
            OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
            LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

            # Train
            CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_train --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
            # Eval
            CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --model_type $MODEL_TYPE --model_name_or_path bert-base-uncased --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
        done
    done
done
