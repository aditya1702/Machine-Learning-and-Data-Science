from text_processer import TextPreprocesser
import pandas as pd
import os
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager
from functools import partial


STORAGE_DIR = '/home/adityavyas17_gmail_com/storage_dir/training_data/'

# Wrapper function to read csv for multiprocessing
def read_csv(filename):
    print(filename)
    return pd.read_csv(STORAGE_DIR + filename)

# Read the data from storage bucket and create a final dataframe
if not 'train.csv' in os.listdir('.'):
    final_df = pd.DataFrame()
    filenames = os.listdir(STORAGE_DIR)
    pool = Pool(processes = os.cpu_count())
    with pool:
        df_list = pool.map(read_csv, filenames)
        final_df = pd.concat(df_list, axis = 0)
    pool.close()
    final_df.to_pickle('train.pkl')

# Read dataframe in chunks to reduce memory usage and faster load times.
dtype_dict = {
    'subreddit': str,
    'subreddit_id': str,
    'title': str,
    'selftext': str,
    'id': str,
    'comment': str,

}
text_processer = TextPreprocesser()
chunk_iterator = pd.read_csv('train.csv', dtype = dtype_dict, chunksize = 100000) # 14590002
for index, chunk in enumerate(chunk_iterator):
    print('################### Chunk = ', str(index + 1), ' ###################')
    try:
        text_processer.preprocess_text(df = chunk)
        if index % 53 == 0:
            text_processer.save_text()
    except:
        continue

# Save the final processed text
text_processer.save_text()
