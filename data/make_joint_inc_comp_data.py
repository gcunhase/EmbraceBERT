import argparse
import os
import csv
from utils import ensure_dir
from collections import defaultdict
from shutil import copyfile

# POS-tag for irrelevant tag selection
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

__author__ = "Gwena Cunha"


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + filename, 'w')
    dict_writer = csv.DictWriter(file_test, fieldnames=keys, delimiter='\t')
    dict_writer.writeheader()
    for data in dict:
        dict_writer.writerow(data)


def make_dataset(root_data_dir, complete_data_dir, incomplete_data_dir, results_dir):
    """
    :param root_data_dir: directory to save data
    :param complete_data_dir: subdirectory with complete data
    :param incomplete_data_dir: subdirectory with incomplete data
    :param results_dir: subdirectory with incomplete data
    :return:
    """
    print("Making incomplete intention classification dataset...")
    complete_data_dir_path = root_data_dir + '/' + complete_data_dir
    incomplete_data_dir_path = root_data_dir + '/' + incomplete_data_dir

    results_dir_path = root_data_dir + '/' + results_dir
    ensure_dir(results_dir_path)

    # Traverse all sub-directories
    complete_files_dir = ['nlu_eval/chatbotcorpus/', 'nlu_eval/askubuntucorpus/', 'nlu_eval/webapplicationscorpus/', 'snips/']
    incomplete_files_dir = ['chatbot/', 'askubuntu/', 'webapplications/', 'snips/']

    # Open train tsv file
    keys = ['sentence', 'label', 'missing', 'target']
    for comp_dir, inc_dir in zip(complete_files_dir, incomplete_files_dir):
        comp_file = comp_dir + 'train.tsv'
        complete_tsv_file = open(complete_data_dir_path + comp_file, 'r')
        reader_complete = csv.reader(complete_tsv_file, delimiter='\t')
        for stt in ['gtts', 'macsay']:
            for tts in ['google', 'sphinx', 'witai']:
                inc_dir_stterror = inc_dir + stt + '_' + tts + '/'

                # Make train file
                inc_file = inc_dir_stterror + 'train.tsv'
                save_path = results_dir_path + '/' + inc_dir_stterror
                ensure_dir(save_path)
                incomplete_tsv_file = open(incomplete_data_dir_path + inc_file, 'r')
                reader_incomplete = csv.reader(incomplete_tsv_file, delimiter='\t')

                row_count = 0
                data_dict = []
                # Copy incomplete data
                for row_inc in reader_incomplete:
                    if row_count != 0:  # skip header
                        # Incomplete
                        data_dict.append({keys[0]: row_inc[0], keys[1]: row_inc[1], keys[2]: row_inc[2], keys[3]: row_inc[3]})
                    row_count += 1

                # Copy complete data, add missing words and target (same as row[0])
                row_count = 0
                for row_comp in reader_complete:
                    if row_count != 0:  # skip header
                        # Complete
                        data_dict.append({keys[0]: row_comp[0], keys[1]: row_comp[1], keys[2]: '', keys[3]: row_comp[0]})
                    row_count += 1

                # Save train in files in the format (sentence, label)
                write_tsv(save_path, 'train.tsv', keys, data_dict)

                # Copy test file from incomplete
                copyfile(incomplete_data_dir_path + inc_dir_stterror + 'test.tsv', save_path + 'test.tsv')

    print("Incomplete intention classification dataset with complete target completed")


def init_args():
    parser = argparse.ArgumentParser(description="Script to make intention recognition dataset")
    parser.add_argument('--root_data_dir', type=str, default="./",
                        help='Directory to save subdirectories, needs to be an absolute path')
    parser.add_argument('--complete_data_dir', type=str, default="intent_processed/",
                        help='Subdirectory with complete data')
    parser.add_argument('--incomplete_data_dir', type=str, default="intent_stterror_data/",
                        help='Subdirectory with incomplete data')
    parser.add_argument('--results_dir', type=str, default="intent_stterror_data_withComplete/",
                        help='Subdirectory to save Joint Complete and Incomplete data')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    make_dataset(args.root_data_dir, args.complete_data_dir, args.incomplete_data_dir, args.results_dir)
