import json
import numpy as np

# Use this script if the model was trained with complete data but you wish to test it with another set of test data,
#  in our case incomplete data.

# root_name = '/media/ceslea/DATA/EmbraceBERT-results-backup/'
root_name = './'
# dataname, model, epoch, bs, tts_stt_type = ["askubuntu", "embracebert_frozenbert", 30, 4, 'gtts_google']
if root_name == './':
    root_name += 'results/test_with_incomplete_results/'

MODEL_ROOT = [
              "{}", #"{}_withDropout0.1", "{}_withDropout0.3",
              #"{}_frozen", "{}_frozen_withDropout0.1", "{}_frozen_withDropout0.3",
              #"embrace{}", "embrace{}_withDropout0.1", "embrace{}_withDropout0.3",
              #"embrace{}_condensed", "embrace{}_condensed_withDropout0.1", "embrace{}_condensed_withDropout0.3",
              #"embrace{}_frozenbert", "embrace{}_frozenbert_withDropout0.1", "embrace{}_frozenbert_withDropout0.3",
              #"embrace{}_frozenbert_condensed", "embrace{}_frozenbert_condensed_withDropout0.1", "embrace{}_frozenbert_condensed_withDropout0.3",
              #"embrace{}_with_branches_sharedWeightsAll", "embrace{}_with_branches_sharedWeightsAll_withDropout0.1",
              #"embrace{}_with_branches_sharedWeightsAll_withDropout0.3", "embrace{}_with_branches_condensed_sharedWeightsAll",
              #"embrace{}_with_branches_condensed_sharedWeightsAll_withDropout0.1",
              #"embrace{}_with_branches_condensed_sharedWeightsAll_withDropout0.3",
              #"embrace{}_with_branches_frozenbert_sharedWeightsAll", "embrace{}_with_branches_frozenbert_sharedWeightsAll_withDropout0.1",
              #"embrace{}_with_branches_frozenbert_sharedWeightsAll_withDropout0.3", "embrace{}_with_branches_frozenbert_condensed_sharedWeightsAll",
              #"embrace{}_with_branches_frozenbert_condensed_sharedWeightsAll_withDropout0.1", "embrace{}_with_branches_frozenbert_condensed_sharedWeightsAll_withDropout0.3",
              #
              #"embrace{}_p_selfattention_condensed",
              #"embrace{}_p_selfattention_pytorch",
              #"embrace{}_p_multiheadattention",
              #"embrace{}_p_multihead_bertselfattention", "embrace{}_p_multihead_bertattention",
              #"embrace{}_p_multihead_bertselfattention_in_p",
              #"embrace{}withkeyvaluequery_p_multinomial",
              #"embrace{}withkeyvaluequery_p_attention_clsquery_weights",
              "{}withatt",
              "{}withattclsprojection",
              "{}withprojection",
              "{}withprojectionatt",
              #"{}withattprojection",
              "embrace{}_attention_p_multinomial",
              "embrace{}_attention_p_attention_clsquery_weights",
              "embrace{}_projection_p_multinomial",
              "embrace{}_projection_p_attention_clsquery_weights",
              "embrace{}concatatt_attention_p_multinomial",
              "embrace{}concatatt_attention_p_attention_clsquery_weights",
              "embrace{}concatatt_projection_p_multinomial",
              "embrace{}concatatt_projection_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequery_attention_p_multinomial",
              "embrace{}withkeyvaluequery_attention_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequery_projection_p_multinomial",
              "embrace{}withkeyvaluequery_projection_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequeryconcatatt_attention_p_multinomial",
              "embrace{}withkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights",
              "embrace{}withkeyvaluequeryconcatatt_projection_p_multinomial",
              "embrace{}withkeyvaluequeryconcatatt_projection_p_attention_clsquery_weights",
]

#MODEL_BERT = []
#for M in MODEL_ROOT:
#    print(M)
#    MODEL_BERT.append(M.format('bert'))
MODEL_ROBERTA = []
for M in MODEL_ROOT:
    MODEL_ROBERTA.append(M.format('roberta'))

MODEL_NAME = {"bert":                                               " BERT-bs{}                              ",
              "bert_withDropout0.1":                                " BERT-bs{}+Dropout0.1                   ",
              "bert_withDropout0.3":                                " BERT-bs{}+Dropout0.3                   ",
              "bertwithatt":                                        " BERTwithAtt-bs{}                       ",
              "bertwithprojection":                                 " BERTwithProjection-bs{}                ",
              "bertwithprojectionatt":                              " BERTwithProjectionAtt-bs{}             ",
              "bertwithattprojection":                              " BERTwithAttProjection-bs{}             ",
              "bertwithattclsprojection":                           " BERTwithAttClsProjection-bs{}          ",
              "bert_frozen":                                        " FrozenBERT-bs{}-ep100                  ",
              "bert_frozen_withDropout0.1":                         " FrozenBERT-bs{}-ep100+Dropout0.1       ",
              "bert_frozen_withDropout0.3":                         " FrozenBERT-bs{}-ep100+Dropout0.3       ",
              "embracebert_attention_p_multinomial":                " EmbraceBERT-bs{}                       ",
              "embracebert_attention_p_attention_clsquery_weights": " EmbraceBERT-bs{}-p_attclsqw            ",
              "embracebert_withDropout0.1":                         " EmbraceBERT-bs{}+Dropout0.1            ",
              "embracebert_withDropout0.3":                         " EmbraceBERT-bs{}+Dropout0.3            ",
              "embracebert_projection_p_multinomial":               " EmbraceBERTwithProj-bs{}               ",
              "embracebert_projection_p_attention_clsquery_weights":" EmbraceBERTwithProj-bs{}-p_attclsqw    ",
              "embracebert_condensed":                              " CondensedEmbraceBERT-bs{}              ",
              "embracebert_condensed_withDropout0.1":               " CondensedEmbraceBERT-bs{}+Dropout0.1   ",
              "embracebert_condensed_withDropout0.3":               " CondensedEmbraceBERT-bs{}+Dropout0.3   ",
              "embracebert_frozenbert":                             " FrozenEBERT-bs{}-ep100                 ",
              "embracebert_frozenbert_withDropout0.1":              " FrozenEBERT-bs{}-ep100+Dropout0.1      ",
              "embracebert_frozenbert_withDropout0.3":              " FrozenEBERT-bs{}-ep100+Dropout0.3      ",
              "embracebert_frozenbert_condensed":                   " FrozenCEBERT-bs{}-ep100                ",
              "embracebert_frozenbert_condensed_withDropout0.1":    " FrozenCEBERT-bs{}-ep100+Dropout0.1     ",
              "embracebert_frozenbert_condensed_withDropout0.3":    " FrozenCEBERT-bs{}-ep100+Dropout0.3     ",
              "embracebert_p_selfattention":                        " EmbraceBERT-bs{}-p_selfatt              ",
              "embracebert_p_selfattention_condensed":              " CondensedEmbraceBERT-bs{}-p_selfatt     ",
              "embracebert_p_selfattention_pytorch":                " EmbraceBERT-bs{}-p_selfatt_pytorch      ",
              "embracebert_p_multiheadattention":                   " EmbraceBERT-bs{}-p_multiheadatt         ",
              "embracebert_p_multihead_bertattention":              " EmbraceBERT-bs{}-p_multihead_bertatt    ",
              "embracebert_p_multihead_bertselfattention":          " EmbraceBERT-bs{}-p_multihead_bertselfatt",
              "embracebert_p_multihead_bertselfattention_in_p":     " EmbraceBERT-bs{}-p_multihead_bertselfatt_in_p",
              "embracebert_p_attention_clsquery":                   " EmbraceBERT-bs{}-p_att_clsquery          ",
              "embracebertwithkeyvaluequery_attention_p_multinomial":                 " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracebertwithkeyvaluequery_attention_p_attention_clsquery_weights":  " EmbraceBERT-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracebertwithkeyvaluequery_projection_p_multinomial":                " EmbraceBERTwithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracebertwithkeyvaluequery_projection_p_attention_clsquery_weights": " EmbraceBERTwithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              "embracebertconcatatt_attention_p_multinomial":                         " EBERTconcatAtt-bs{}                   ",
              "embracebertconcatatt_attention_p_attention_clsquery_weights":          " EBERTconcatAtt-bs{}-p_attclsqw        ",
              "embracebertconcatatt_projection_p_multinomial":                        " EBERTconcatAttwithProj-bs{}           ",
              "embracebertconcatatt_projection_p_attention_clsquery_weights":         " EBERTconcatAttwithProj-bs{}-p_attclsqw",
              "embracebertwithkeyvaluequeryconcatatt_attention_p_multinomial":                 " EmbraceBERTconcatatt-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracebertwithkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights":  " EmbraceBERTconcatatt-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracebertwithkeyvaluequeryconcatatt_projection_p_multinomial":                " EmbraceBERTconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracebertwithkeyvaluequeryconcatatt_projection_p_attention_clsquery_weights": " EmbraceBERTconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              "embracebert_with_branches_sharedWeightsAll":                                      " EmbraceBERT-bs{}+Branches                     ",
              "embracebert_with_branches_sharedWeightsAll_withDropout0.1":                       " EmbraceBERT-bs{}+Branches+Dropout0.1          ",
              "embracebert_with_branches_sharedWeightsAll_withDropout0.3":                       " EmbraceBERT-bs{}+Branches+Dropout0.3          ",
              "embracebert_with_branches_condensed_sharedWeightsAll":                            " CondensedEmbraceBERT-bs{}+Branches            ",
              "embracebert_with_branches_condensed_sharedWeightsAll_withDropout0.1":             " CondensedEmbraceBERT-bs{}+Branches+Dropout0.1 ",
              "embracebert_with_branches_condensed_sharedWeightsAll_withDropout0.3":             " CondensedEmbraceBERT-bs{}+Branches+Dropout0.3 ",
              "embracebert_with_branches_frozenbert_sharedWeightsAll":                           " FrozenEmbraceBERT-bs{}+Branches               ",
              "embracebert_with_branches_frozenbert_sharedWeightsAll_withDropout0.1":            " FrozenEmbraceBERT-bs{}+Branches+Dropout0.1    ",
              "embracebert_with_branches_frozenbert_sharedWeightsAll_withDropout0.3":            " FrozenEmbraceBERT-bs{}+Branches+Dropout0.3    ",
              "embracebert_with_branches_frozenbert_condensed_sharedWeightsAll":                 " FrozenCEBERT-bs{}+Branches                    ",
              "embracebert_with_branches_frozenbert_condensed_sharedWeightsAll_withDropout0.1":  " FrozenCEBERT-bs{}+Branches+Dropout0.1         ",
              "embracebert_with_branches_frozenbert_condensed_sharedWeightsAll_withDropout0.3":  " FrozenCEBERT-bs{}+Branches+Dropout0.3         ",
              "roberta":                     " RoBERTa-bs{}                     ",
              "robertawithatt":              " RoBERTawithAtt-bs{}              ",
              "robertawithprojection":       " RoBERTawithProjection-bs{}       ",
              "robertawithprojectionatt":    " RoBERTawithProjectionAtt-bs{}    ",
              "robertawithattclsprojection": " RoBERTawithAttClsProjection-bs{} ",
              "embraceroberta": " EmbraceRoBERTa-bs{}                        ",
              "embraceroberta_attention_p_multinomial": " EmbraceRoBERTa-bs{}                        ",
              "embraceroberta_attention_p_attention_clsquery_weights": " EmbraceRoBERTa-bs{}-p_att_clsquery_weights ",
              "embraceroberta_projection_p_multinomial": " EmbraceRoBERTawithProj-bs{}                ",
              "embraceroberta_projection_p_attention_clsquery_weights": " EmbraceRoBERTawithProj-bs{}-p_attclsqw     ",
              "embracerobertawithkeyvaluequery_attention_p_multinomial": " EmbraceRoBERTa-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracerobertawithkeyvaluequery_attention_p_attention_clsquery_weights": " EmbraceRoBERTa-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracerobertawithkeyvaluequery_projection_p_multinomial": " EmbraceRoBERTawithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracerobertawithkeyvaluequery_projection_p_attention_clsquery_weights": " EmbraceRoBERTawithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              "embracerobertaconcatatt_attention_p_multinomial": " ERoBERTaconcatAtt-bs8                  ",
              "embracerobertaconcatatt_attention_p_attention_clsquery_weights": " ERoBERTaTconcatAtt-bs8_attclsqw         ",
              "embracerobertaconcatatt_projection_p_multinomial": " ERoBERTaconcatAttwithProj-bs8          ",
              "embracerobertaconcatatt_projection_p_attention_clsquery_weights": " ERoBERTaconcatAttwithProj-bs8_attclsqw ",
              "embracerobertawithkeyvaluequeryconcatatt_attention_p_multinomial": " ERoBERTaconcatatt-bs{}-p_multiheadatt_bertKeyValQuery                  ",
              "embracerobertawithkeyvaluequeryconcatatt_attention_p_attention_clsquery_weights": " ERoBERTaconcatatt-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw         ",
              "embracerobertawithkeyvaluequeryconcatatt_projection_p_multinomial": " ERoBERTaconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery          ",
              "embracerobertawithkeyvaluequeryconcatatt_projection_p_attention_clsquery_weights": " ERoBERTaconcatattWithProj-bs{}-p_multiheadatt_bertKeyValQuery_attclsqw ",
              }

for dataname in ["chatbot"]:  #["askubuntu", "chatbot", "webapplications", "snips"]:
    if dataname == "snips":
        bs_array = [32]
        epoch_array = [100]
    else:
        bs_array = [8]  #, 8] #, 16]
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
                for model_type in [MODEL_ROBERTA]:
                    for bs in bs_array:
                        print("| ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                        for model in model_type:
                            model_name = MODEL_NAME[model]

                            # print("{dataname} {model} - ep{epoch} bs{bs}".format(dataname=dataname, model=model, epoch=epoch, bs=bs))

                            root_dir = '{root_name}{model}/{dataname}/complete/{dataname}_ep{epoch}_bs{bs}_'.\
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
