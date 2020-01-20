#!/bin/bash -v

MODEL_TYPE=embracebert
# MODEL_TYPE=embraceroberta

if [ $MODEL_TYPE == "embracebert" ]; then
  MODEL_NAME_OR_PATH="bert-base-uncased"
else
  MODEL_NAME_OR_PATH="roberta-base"
fi

echo $MODEL_NAME_OR_PATH