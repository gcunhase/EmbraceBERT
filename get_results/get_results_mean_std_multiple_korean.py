import json
import numpy as np

# Use this script if you want to obtain the mean and std of models trained and tested with incomplete data

"""
root_name = './results/'

MODEL_ROOT = [
              "{}", "{}_withDropout0.1",
              "embrace{}",
              "embrace{}_withDropout0.1"
]
"""

#root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = './results_korean/'
MODEL_ROOT = [
              #"{}",
              #"{}withatt", "{}withattclsprojection",
              #"{}withprojection", "{}withprojectionatt",
              #"embrace{}_attention_p_multinomial",
              #"embrace{}_attention_p_attention_clsquery_weights",
              #"embrace{}_projection_p_multinomial",
              #"embrace{}_projection_p_attention_clsquery_weights",
              #"embrace{}concatatt_attention_p_multinomial",
              #"embrace{}concatatt_attention_p_attention_clsquery_weights",
              #"embrace{}concatatt_projection_p_multinomial",
              #"embrace{}concatatt_projection_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequery_attention_p_multinomial",
              "embrace{}withkeyvaluequery_attention_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequery_projection_p_multinomial",
              "embrace{}withkeyvaluequery_projection_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequeryconcatatt_attention_p_multinomial",
              "embrace{}withkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequeryconcatatt_projection_p_multinomial",
              "embrace{}withkeyvaluequeryconcatatt_projection_p_attention_clsquery_weights",
]

MODEL_BERT = []
for M in MODEL_ROOT:
    MODEL_BERT.append(M.format('bert'))
#MODEL_ROBERTA = []
#for M in MODEL_ROOT:
#    MODEL_ROBERTA.append(M.format('roberta'))

MODEL_NAME = {"bert":                                            " BERT-bs{}                               ",
              "bertwithatt":                                     " BERTwithAtt-bs{}                        ",
              "bertwithprojection":                              " BERTwithProjection-bs{}                 ",
              "bertwithprojectionatt":                           " BERTwithProjectionAtt-bs{}              ",
              "bertwithattprojection":                           " BERTwithAttProjection-bs{}              ",
              "bertwithattclsprojection":                        " BERTwithAttClsProjection-bs{}           ",
              "embracebert":                                        " EmbraceBERT-bs{}                        ",
              "embracebert_withDropout0.1":                         " EmbraceBERT-bs{}+Dropout0.1             ",
              "embracebert_attention_p_multinomial":                " EmbraceBERT-bs{}                        ",
              "embracebert_attention_p_attention_clsquery_weights": " EmbraceBERT-bs{}-p_attclsqw             ",
              "embracebert_projection_p_multinomial":               " EmbraceBERTwithProj-bs{}                ",
              "embracebert_projection_p_attention_clsquery_weights":" EmbraceBERTwithProj-bs{}-p_attclsqw     ",
              "embracebertwithkeyvaluequery_p_multinomial":                           " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracebertwithkeyvaluequery_p_selfattention":                         " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery_selfatt          ",
              "embracebertwithkeyvaluequery_attention_p_multinomial":                 " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracebertwithkeyvaluequery_attention_p_attention_clsquery_weights":  " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracebertwithkeyvaluequery_projection_p_multinomial":                " EmbraceBERTwithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracebertwithkeyvaluequery_projection_p_attention_clsquery_weights": " EmbraceBERTwithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              "embracebertconcatatt_attention_p_multinomial":                         " EBERTconcatAtt-bs8                     ",
              "embracebertconcatatt_attention_p_attention_clsquery_weights":          " EBERTconcatAtt-bs8_attclsqw            ",
              "embracebertconcatatt_projection_p_multinomial":                        " EBERTconcatAttwithProj-bs8             ",
              "embracebertconcatatt_projection_p_attention_clsquery_weights":         " EBERTconcatAttwithProj-bs8_attclsqw    ",
              "embracebertwithkeyvaluequeryconcatatt_attention_p_multinomial":                 " EmbraceBERTconcatatt-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracebertwithkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights":  " EmbraceBERTconcatatt-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracebertwithkeyvaluequeryconcatatt_projection_p_multinomial":                " EmbraceBERTconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracebertwithkeyvaluequeryconcatatt_projection_p_attention_clsquery_weights": " EmbraceBERTconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              }

is_comp_inc = False
for dataname in ["chatbot"]:  #["askubuntu", "chatbot", "webapplications", "snips"]:
    if dataname == "snips":
        bs_array = [16, 32]
        epoch_array = [3]
    else:
        bs_array = [8] #, 16]
        epoch_array = [100]

    for epoch in epoch_array:
        print("- {}, ep {}".format(dataname.upper(), epoch))
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

                            prefix = "stterror_withComplete" if is_comp_inc else "stterror"
                            root_dir = '{root_name}{model}/{dataname}/{prefix}/{tts_stt_type}/{dataname}_ep{epoch}_bs{bs}_'.\
                                format(root_name=root_name, model=model, dataname=dataname, prefix=prefix, epoch=epoch, bs=bs, tts_stt_type=tts_stt_type)
                            f1_micro_str_all = ""
                            for perc in [0.1]:
                                f1_micro_arr = []
                                if bs == 4:
                                    f1_micro_str_all += "|{} ".format(model_name.format(bs))
                                else:
                                    f1_micro_str_all += "|{}".format(model_name.format(bs))
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
