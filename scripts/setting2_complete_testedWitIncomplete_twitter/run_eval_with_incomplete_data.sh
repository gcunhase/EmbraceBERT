#!/bin/bash -v

MODEL_TYPE=bertwithatt  # Options = [bert, bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]
MODEL_NAME="${MODEL_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results_twitter/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter/${MODEL_NAME}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_twitter_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=0
BS_EVAL=1
for BS_TRAIN in 4 8; do
  for EPOCH in 100; do
      for DATASET in "sentiment140"; do
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

              # Eval: test with noisy data (original tweet)
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/"
              EVAL_OUTPUT_FILENAME="eval_results_inc"
              CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME

              # Eval: test with noisy and clean data -> NOT NEEDED (same as INC)
              #DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_inc_with_corr_sentences/"
              #EVAL_OUTPUT_FILENAME="eval_results_inc_comp"
              #CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
          done
      done
  done
done

MODEL_TYPE=bertwithprojection  # Options = [bert, bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]
MODEL_NAME="${MODEL_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results_twitter/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter/${MODEL_NAME}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_twitter_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=0
BS_EVAL=1
for BS_TRAIN in 4 8; do
  for EPOCH in 100; do
      for DATASET in "sentiment140"; do
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

              # Eval: test with noisy data (original tweet)
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/"
              EVAL_OUTPUT_FILENAME="eval_results_inc"
              CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME

              # Eval: test with noisy and clean data -> NOT NEEDED (same as INC)
              #DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_inc_with_corr_sentences/"
              #EVAL_OUTPUT_FILENAME="eval_results_inc_comp"
              #CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
          done
      done
  done
done

MODEL_TYPE=bertwithattclsprojection  # Options = [bert, bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]
MODEL_NAME="${MODEL_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results_twitter/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter/${MODEL_NAME}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_twitter_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=0
BS_EVAL=1
for BS_TRAIN in 4 8; do
  for EPOCH in 100; do
      for DATASET in "sentiment140"; do
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

              # Eval: test with noisy data (original tweet)
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/"
              EVAL_OUTPUT_FILENAME="eval_results_inc"
              CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME

              # Eval: test with noisy and clean data -> NOT NEEDED (same as INC)
              #DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_inc_with_corr_sentences/"
              #EVAL_OUTPUT_FILENAME="eval_results_inc_comp"
              #CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
          done
      done
  done
done

MODEL_TYPE=bertwithprojectionatt  # Options = [bert, bertwithatt, bertwithprojection, bertwithprojectionatt, bertwithattclsprojection]
MODEL_NAME="${MODEL_TYPE}"

LANGUAGE="english"  # Options = [english, korean]
if [[ $LANGUAGE == *"english"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
  OUTPUT_DIR="../../results_twitter/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter/${MODEL_NAME}/"
  DATA_PATH_NAME="twitter_sentiment_data"
else
  MODEL_NAME_OR_PATH="bert-base-multilingual-uncased"
  OUTPUT_DIR="../../results_twitter_korean/${MODEL_NAME}/"
  EVAL_DIR="../../results_twitter_korean/test_with_incomplete_results/${MODEL_NAME}/"
  mkdir $EVAL_DIR
  RUN_DIR="../../runs_twitter_korean/${MODEL_NAME}/"
  DATA_PATH_NAME="korean_twitter_sentiment_data"
fi
echo $MODEL_NAME_OR_PATH

CUDA_ID=0
BS_EVAL=1
for BS_TRAIN in 4 8; do
  for EPOCH in 100; do
      for DATASET in "sentiment140"; do
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

              # Eval: test with noisy data (original tweet)
              DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}/"
              EVAL_OUTPUT_FILENAME="eval_results_inc"
              CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME

              # Eval: test with noisy and clean data -> NOT NEEDED (same as INC)
              #DATA_DIR="../../data/${DATA_PATH_NAME}/${DATASET}_inc_with_corr_sentences/"
              #EVAL_OUTPUT_FILENAME="eval_results_inc_comp"
              #CUDA_VISIBLE_DEVICES=$CUDA_ID python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --num_train_epochs_frozen_bert 100.0 --logging_steps 1 --do_eval --do_lower_case --data_dir $DATA_DIR --max_seq_length 128 --per_gpu_eval_batch_size=$BS_EVAL --per_gpu_train_batch_size=$BS_TRAIN --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH --overwrite_output_dir --overwrite_cache --save_best --log_dir $LOG_DIR_PATH --eval_type "incomplete_test" --eval_output_dir $EVAL_PATH --eval_output_filename $EVAL_OUTPUT_FILENAME
          done
      done
  done
done