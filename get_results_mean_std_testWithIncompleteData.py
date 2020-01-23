import json
import numpy as np

# Use this script if the model was trained with complete data but you wish to test it with another set of test data,
#  in our case incomplete data.

# root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = './'
# dataname, model, epoch, bs, tts_stt_type = ["askubuntu", "embracebert_frozenbert", 30, 4, 'gtts_google']
if root_name == './':
    root_name += 'results/test_with_incomplete_results/'

MODEL = ["bert", "embracebert", "embracebert_condensed", "bert_frozen", "embracebert_frozenbert",
         "roberta", "embraceroberta", "embraceroberta_condensed", "roberta_frozen", "embraceroberta_frozenbert"]
MODEL_NAME = {"bert": " BERT-bs{}                ",
              "embracebert": " EmbraceBERT-bs{}         ",
              "embracebert_condensed": " CondensedEmbraceBERT-bs{}",
              "bert_frozen": " FrozenBERT-bs{}-ep100    ",
              "embracebert_frozenbert": " FrozenEBERT-bs{}-ep100   ",
              "roberta": " RoBERTa-bs{}             ",
              "embraceroberta": " EmbraceRoBERTa-bs{}      ",
              "embraceroberta_condensed": " CondensedEmbraceRoBERTa-bs{}",
              "roberta_frozen": " FrozenRoBERTa-bs{}-ep100    ",
              "embraceroberta_frozenbert": " FrozenERoBERTa-bs{}-ep100   "
              }

for dataname in ["snips"]:  #, "askubuntu", "chatbot", "webapplications", "snips"]:
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

                for bs in bs_array:
                    print("-----------------------------------------")
                    for model in MODEL:
                        model_name = MODEL_NAME[model]

                        # print("{dataname} {model} - ep{epoch} bs{bs}".format(dataname=dataname, model=model, epoch=epoch, bs=bs))

                        root_dir = '{root_name}{model}/{dataname}/{dataname}_ep{epoch}_bs{bs}_'.\
                            format(root_name=root_name, model=model, dataname=dataname, epoch=epoch, bs=bs)

                        f1_micro_str_all = ""
                        for perc in [0.1]:
                            f1_micro_arr = []
                            if bs == 4:
                                f1_micro_str_all += "|{} ".format(model_name.format(bs))
                            else:
                                f1_micro_str_all += "|{}".format(model_name.format(bs))
                            for i in range(1, 10 + 1):
                                tmp_dir = "{}seed{}/".format(root_dir, i)
                                tmp_dir += "eval_results_{}.json".format(tts_stt_type)

                                # Load json file
                                with open(tmp_dir, 'r') as f:
                                    datastore = json.load(f)
                                    f1_score = datastore['f1']
                                    f1_micro_arr.append(f1_score)
                                    f1_micro_str_all += "|{:.2f}".format(f1_score*100)

                            f1_micro_str_all += "|{:.2f}|{:.2f}|".format(np.mean(f1_micro_arr)*100, np.std(f1_micro_arr)*100)

                        print(f1_micro_str_all)
