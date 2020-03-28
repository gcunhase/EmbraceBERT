#!/bin/bash -v

MODEL_TYPE=embraceroberta
# MODEL_TYPE=embraceroberta
IS_FROZEN=false

if [ "$IS_FROZEN" = true ]; then
  echo "FROZEN"
else
  echo "NOT FROZEN"
fi

if [[ $MODEL_TYPE = "bert" || $MODEL_TYPE = "roberta" ]]; then
  echo "BERT"
else
  echo "NOT BERT"
fi

if [[ $MODEL_TYPE == *"bert"* ]]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
else
  MODEL_NAME_OR_PATH="roberta-base"
fi

echo $MODEL_NAME_OR_PATH