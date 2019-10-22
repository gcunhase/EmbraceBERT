import json
import numpy as np


# Complete
# AskUbuntu
root_dir = 'results/embracebert/askubuntu/askubuntu_complete_ep100_bs4_'
# root_dir = '/media/ceslea/RESEARCH/stacked_debert/IntentClassifier-RoBERTa/results/results_complete_earlyStopWithEvalLoss/bert/askubuntu_ep3/askubuntu_bert_ep3_bs4_'

f1_micro_str_all = ""
for perc in [0.1]:
    f1_micro_arr = []
    f1_micro_str_all += "| {}    ".format(0)
    for i in range(1, 10 + 1):
        tmp_dir = "{}seed{}/".format(root_dir, i)
        # tmp_dir = "{}seed{}_second_layer_epae1000/".format(root_dir, i)
        # tmp_dir += "eval_results_test.json"
        tmp_dir += "eval_results.json"

        # Load json file
        with open(tmp_dir, 'r') as f:
            datastore = json.load(f)
            f1_score = datastore['f1']
            f1_micro_arr.append(f1_score)
            f1_micro_str_all += "|{:.2f}".format(f1_score*100)

    f1_micro_str_all += "|{:.2f}|{:.2f}|\n".format(np.mean(f1_micro_arr)*100, np.std(f1_micro_arr)*100)

print(f1_micro_str_all)
