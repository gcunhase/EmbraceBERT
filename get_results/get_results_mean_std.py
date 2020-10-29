import json
import numpy as np


# STTError
# Parameters:
#   is_incomplete_test: True if model was trained with complete data and tested with incomplete

#root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = '../results/'
# stt_error, dataname, model, epoch, bs, tts_stt_type = [False, "snips", "embracebert_with_branches_frozenbert_condensed_sharedWeightsAll", 3, 32, 'gtts_google']
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "chatbot", "embracebertconcatatt_attention_p_attention_clsquery_weights", 100, 3, 8, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "embracebertwithkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights", 100, 3, 32, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "chatbot", "embracebertwithkeyvaluequeryconcatatt_attention_p_multinomial_withSoftmax", 100, 3, 8, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "embracebertwithkeyvaluequeryconcatatt_projection_p_multinomial", 100, 3, 32, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "embracebertwithkeyvaluequery_projection_p_attention_clsquery_weights", 100, 3, 32, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "embracebertconcatatt_projection_p_attention_clsquery_weights", 100, 3, 32, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "embracebertconcatatt_attention_p_attention_clsquery_weights", 100, 3, 32, 'gtts_google']  # _withDropout0.1
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "bertwithatt", 100, 3, 48, 'gtts_google']  # _withDropout0.1

# Setting 3: trained and tested with incomplete data
stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [False, "snips", "bertwithatt", 100, 3, 48, 'macsay_sphinx']
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [True, "snips", "embracebert_attention_p_attention_clsquery_weights", 100, 3, 32, 'macsay_witai']
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [True, "snips", "embracebertwithkeyvaluequery_attention_p_attention_clsquery_weights", 100, 3, 32, 'macsay_sphinx']
#stt_error, dataname, model, epoch, epoch_q, bs, tts_stt_type = [True, "snips", "embracebert_projection_p_attention_clsquery_weights", 100, 3, 32, 'macsay_sphinx']


if root_name == './':
    root_name += 'results/backedup/'

if stt_error:
    if epoch_q > 3:
        root_dir = '{root_name}{model}/{dataname}/stterror/{tts_stt_type}/{dataname}_ep{epoch}_epQ{epoch_q}_bs{bs}_'. \
            format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, epoch_q=epoch_q, bs=bs, tts_stt_type=tts_stt_type)
    else:
        root_dir = '{root_name}{model}/{dataname}/stterror/{tts_stt_type}/{dataname}_ep{epoch}_bs{bs}_'.\
            format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, bs=bs, tts_stt_type=tts_stt_type)
else:
    root_dir = '{root_name}{model}/{dataname}/complete/{dataname}_ep{epoch}_bs{bs}_'.\
        format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, bs=bs)

print(model)
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
