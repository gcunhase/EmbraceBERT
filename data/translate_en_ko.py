# pip install googletrans
import argparse
from utils import ensure_dir
import csv
from googletrans import Translator


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + filename, 'w')
    dict_writer = csv.DictWriter(file_test, fieldnames=keys, delimiter='\t')
    dict_writer.writeheader()
    for data in dict:
        dict_writer.writerow(data)


def translate_file(src_data_dir_path, dest_data_dir_path, src_dir, keys, src_file_type='train.tsv'):

    translator = Translator()

    src_file = src_data_dir_path + src_dir + src_file_type
    save_path = dest_data_dir_path + '/' + src_dir
    ensure_dir(save_path)

    src_tsv_file = open(src_file, 'r')
    reader_src = csv.reader(src_tsv_file, delimiter='\t')
    print(src_file)

    row_count = 0
    data_dict = []
    # Translate source data
    for row_src in reader_src:
        if row_count != 0:  # skip header
            # Source
            src_text = row_src[0]
            if len(src_text) > 1:
                dest_text = translator.translate(src_text, src=args.src_language, dest=args.dest_language).text
            else:
                dest_text = src_text
            if len(keys) == 2:
                data_dict.append(
                    {keys[0]: dest_text, keys[1]: row_src[1]})
            else:
                src_missing_word = row_src[2]
                if len(src_missing_word) > 1: # " " or ""
                    dest_missing_word = translator.translate(src_missing_word, src=args.src_language, dest=args.dest_language).text
                else:
                    dest_missing_word = src_missing_word
                src_target = row_src[3]
                if len(src_target) > 1:
                    dest_target = translator.translate(src_target, src=args.src_language, dest=args.dest_language).text
                else:
                    dest_target = src_target
                data_dict.append({keys[0]: dest_text, keys[1]: row_src[1], keys[2]: dest_missing_word, keys[3]: dest_target})
        row_count += 1

    # Save train in files in the format (sentence, label)
    write_tsv(save_path, src_file_type, keys, data_dict)


def translate_dataset(args):

    print("Translating dataset...")
    src_data_dir_path = args.root_data_dir + '/' + args.src_data_dir
    dest_data_dir_path = args.root_data_dir + '/' + args.dest_data_dir
    ensure_dir(dest_data_dir_path)


    # Traverse all sub-directories
    if 'intent_processed' in args.src_data_dir:
        data_subdir = ['nlu_eval/askubuntucorpus/', 'nlu_eval/chatbotcorpus/', 'nlu_eval/webapplicationscorpus/', 'snips/']

        keys = ['sentence', 'label']
        # Open train tsv file
        for src_dir in data_subdir:
            translate_file(src_data_dir_path, dest_data_dir_path, src_dir, keys, src_file_type='train.tsv')
            translate_file(src_data_dir_path, dest_data_dir_path, src_dir, keys, src_file_type='test.tsv')

    else:
        data_subdir = ['askubuntu/', 'chatbot/', 'webapplications/', 'snips/']

        keys = ['sentence', 'label', 'missing', 'target']
        # Open train tsv file
        for src_dir in data_subdir:
            for stt in ['gtts', 'macsay']:
                for tts in ['google', 'sphinx', 'witai']:
                    translate_file(src_data_dir_path, dest_data_dir_path, src_dir+stt+'_'+tts+'/', keys, src_file_type='train.tsv')
                    translate_file(src_data_dir_path, dest_data_dir_path, src_dir+stt+'_'+tts+'/', keys, src_file_type='test.tsv')


def init_args():
    parser = argparse.ArgumentParser(description="Script to make intention recognition dataset")
    parser.add_argument('--root_data_dir', type=str, default="./",
                        help='Directory to save subdirectories, needs to be an absolute path')
    parser.add_argument('--src_language', type=str, default="en", help='Source language: english')
    parser.add_argument('--src_data_dir', type=str, default="intent_processed/", help='Subdirectory with source data')
    # parser.add_argument('--src_data_dir', type=str, default="intent_stterror_data/", help='Subdirectory with source data')
    parser.add_argument('--dest_language', type=str, default="ko", help='Target language: korean')
    parser.add_argument('--dest_data_dir', type=str, default="korean_intent_processed/", help='Subdirectory with target data')
    # parser.add_argument('--dest_data_dir', type=str, default="korean_intent_stterror_data/", help='Subdirectory with target data')
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    translate_dataset(args)
