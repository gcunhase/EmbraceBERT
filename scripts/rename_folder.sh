FOLDER_PATH="/media/ceslea/RESEARCH/PycharmProjects/EmbraceBERT/results/embracebert_withDropout0.3/askubuntu/complete"

BS=4
for EP in 30 100; do
  for SEED in 1 2 3 4 5 6 7 8 9 10; do
      mv "${FOLDER_PATH}/askubuntu_ep${EP}_bs${BS}_dropout0.3_seed${SEED}" "${FOLDER_PATH}/askubuntu_ep${EP}_bs${BS}_seed${SEED}"
  done
done