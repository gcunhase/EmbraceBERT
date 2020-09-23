import csv


INTENTION_TAGS = {
    'snips': {'AddToPlaylist': 0,
              'BookRestaurant': 1,
              'GetWeather': 2,
              'PlayMusic': 3,
              'RateBook': 4,
              'SearchCreativeWork': 5,
              'SearchScreeningEvent': 6},
    'ChatbotCorpus': {'DepartureTime': 0,
                      'FindConnection': 1},
    'AskUbuntuCorpus': {'Make Update': 0,
                        'Setup Printer': 1,
                        'Shutdown Computer': 2,
                        'Software Recommendation': 3,
                        'None': 4},
    'WebApplicationsCorpus': {'Change Password': 0,
                              'Delete Account': 1,
                              'Download Video': 2,
                              'Export Data': 3,
                              'Filter Spam': 4,
                              'Find Alternative': 5,
                              'Sync Accounts': 6,
                              'None': 7}
}


SENTIMENT_TAGS = {'Sentiment140': {'Negative': 0,
                                   'Positive': 1}
                  }


def write_tsv(intention_dir_path, filename, keys, dict):
    file_test = open(intention_dir_path + "/" + filename, 'wt')
    dict_writer = csv.writer(file_test, delimiter='\t')
    dict_writer.writerow(keys)
    r = zip(*dict.values())
    for d in r:
        dict_writer.writerow(d)
