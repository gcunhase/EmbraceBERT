#!/bin/bash -v

MODEL_TYPE=bertwithatt  # Options=[bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4; do
  for DATASET in "sentiment140"; do
      echo $DATASET
      for EPOCH in 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_corrected_sentences/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="twitter/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
  done
done


MODEL_TYPE=bertwithprojection  # Options=[bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4; do
  for DATASET in "sentiment140"; do
      echo $DATASET
      for EPOCH in 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_corrected_sentences/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="twitter/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
  done
done

MODEL_TYPE=bertwithprojectionatt  # Options=[bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4; do
  for DATASET in "sentiment140"; do
      echo $DATASET
      for EPOCH in 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_corrected_sentences/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="twitter/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
  done
done

MODEL_TYPE=bertwithattclsprojection  # Options=[bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results/${MODEL_TYPE}/"
  RUN_DIR="../../runs/${MODEL_TYPE}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_korean/${MODEL_TYPE}/"
  RUN_DIR="../../runs_korean/${MODEL_TYPE}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

BS_EVAL=1
for BS_TRAIN in 4; do
  for DATASET in "sentiment140"; do
      echo $DATASET
      for EPOCH in 100; do
          echo "Training ${DATASET} dataset with complete data for ${EPOCH} epochs"

          DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_corrected_sentences/"

          for SEED in 1 2 3 4 5 6 7 8 9 10; do
              RESULT_DIR="twitter/complete/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}"
              OUT_PATH="${OUTPUT_DIR}/${RESULT_DIR}"
              LOG_DIR_PATH="${RUN_DIR}/${RESULT_DIR}"

              # Train
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_train --evaluate_during_training --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
              # Eval
              CUDA_VISIBLE_DEVICES=0 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH
          done
      done
  done
done
