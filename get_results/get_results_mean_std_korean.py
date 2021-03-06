import json
import numpy as np


# STTError
# Parameters:
#   is_incomplete_test: True if model was trained with complete data and tested with incomplete

#root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = './results_korean/'
stt_error, dataname, model, epoch, bs, tts_stt_type = [False, "chatbot", "bert", 100, 8, 'gtts_google']  # _withDropout0.1
# root_name = './results_twitter/'
# stt_error, dataname, model, epoch, bs, tts_stt_type = [False, "sentiment140", "bertwithatt", 100, 4, 'gtts_google']  # _withDropout0.1

if stt_error:
    root_dir = '{root_name}{model}/{dataname}/stterror_withComplete/{tts_stt_type}/{dataname}_ep{epoch}_bs{bs}_'.\
        format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, bs=bs, tts_stt_type=tts_stt_type)
else:
    root_dir = '{root_name}{model}/{dataname}/complete/{dataname}_ep{epoch}_bs{bs}_'.\
        format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, bs=bs)

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
