import json
import numpy as np

# Use this script if the model was trained with complete data but you wish to test it with another set of test data,
#  in our case incomplete data.

#root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = './results/'

MODEL_ROOT = [#"embrace{}withquery_p_multinomial",
              #"embrace{}withquery_bertKeyValue_p_multinomial",
              #"embrace{}withquery_p_multinomial_epq0",
              "embrace{}withquery_p_multinomial_withDropout0.1"
]

MODEL_BERT = []
for M in MODEL_ROOT:
    MODEL_BERT.append(M.format('bert'))
#MODEL_ROBERTA = []
#for M in MODEL_ROOT:
#    MODEL_ROBERTA.append(M.format('roberta'))

MODEL_NAME = {"embracebertwithquery_p_multinomial":                " EmbraceBERT-bs{}-p_multiheadatt_bertquery_epQ{}  ",
              "embracebertwithquery_bertKeyValue_p_multinomial":   " EmbraceBERT-bs{}-p_multiheadatt_bertKeyVal_epQ{} ",
              "embracebertwithquery_p_multinomial_epq0":           " EmbraceBERT-bs{}-p_multiheadatt_bertquery_epQ{}  ",
              "embracebertwithquery_p_multinomial_withDropout0.1": " EmbraceBERT-bs{}-p_multiheadatt_bertquery_epQ{}+Dropout0.1",
              }

is_comp_inc = True
for dataname in ["webapplications"]:  #["askubuntu", "chatbot", "webapplications", "snips"]:
    if dataname == "snips":
        bs_array = [16, 32]
        epoch_array = [3]
    else:
        bs_array = [4, 16]
        epoch_array = [100]

    for epoch in epoch_array:
        print("- {} - ep{}".format(dataname.upper(), epoch))
        for tts in ["gtts", "macsay"]:
            for stt in ["google", "sphinx", "witai"]:
                tts_stt_type = tts + "_" + stt
                print(tts_stt_type)

                #for bs in bs_array:
                #    print("-----------------------------------------")
                #for model_type in [MODEL_BERT, MODEL_ROBERTA]:
                for model_type in [MODEL_BERT]:
                    for bs in bs_array:
                        print("| ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                        for model in model_type:
                            model_name = MODEL_NAME[model]
                            for epq in [3]: #, 10]:

                                # print("{dataname} {model} - ep{epoch} bs{bs}".format(dataname=dataname, model=model, epoch=epoch, bs=bs))
                                prefix = "stterror_withComplete" if is_comp_inc else "stterror"
                                if epq == 3:
                                    root_dir = '{root_name}{model}/{dataname}/{prefix}/{tts_stt_type}/{dataname}_ep{epoch}_bs{bs}_'.\
                                        format(root_name=root_name, model=model, dataname=dataname, prefix=prefix, epoch=epoch, bs=bs, tts_stt_type=tts_stt_type)
                                else:
                                    root_dir = '{root_name}{model}/{dataname}/{prefix}/{tts_stt_type}/{dataname}_ep{epoch}_epQ{epq}_bs{bs}_'. \
                                        format(root_name=root_name, model=model, dataname=dataname, prefix=prefix, epoch=epoch, epq=epq, bs=bs, tts_stt_type=tts_stt_type)
                                f1_micro_str_all = ""
                                for perc in [0.1]:
                                    f1_micro_arr = []
                                    if bs == 4:
                                        f1_micro_str_all += "|{} ".format(model_name.format(bs, epq))
                                    else:
                                        f1_micro_str_all += "|{}".format(model_name.format(bs, epq))
                                    for i in range(1, 10 + 1):
                                        tmp_dir = "{}seed{}/".format(root_dir, i)
                                        tmp_dir += "eval_results.json"

                                        # Load json file
                                        with open(tmp_dir, 'r') as f:
                                            datastore = json.load(f)
                                            f1_score = datastore['f1']
                                            f1_micro_arr.append(f1_score)
                                            f1_micro_str_all += "|{:.2f}".format(f1_score*100)

                                    f1_micro_str_all += "|{:.2f}|{:.2f}|".format(np.mean(f1_micro_arr)*100, np.std(f1_micro_arr)*100)

                                print(f1_micro_str_all)
