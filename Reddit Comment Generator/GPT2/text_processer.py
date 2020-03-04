import os
from multiprocessing import Pool, Manager
from functools import partial
from ctypes import c_char_p


class TextPreprocesser(object):
  def __init__(self):
    manager = Manager()
    self.post_texts_with_comments = manager.dict()

  def _multi_preprocess(self, post_id):
      try:
        temp_df = self.df[self.df['id'] == post_id]
        post_text = str(temp_df['title'].unique()[0])
        subreddit_id = temp_df['subreddit_id'].unique()[0]
        comments = temp_df['comment'].values[:10]
        self_text = str(temp_df['selftext'].unique()[0])
        self_text = self_text if len(self_text) >= 3 else ''

        # Remove empty, deleted and removed posts
        if post_text in {'', '[deleted]', '[removed]'}:
          return
        post_text = '****S ' + str(subreddit_id) + '\n' + str(post_text) + '\n\n' + str(self_text) + '\n' + '****ES ' + post_id

        # Process comments for the post
        comments_text = ''
        for comment in comments:

          # Remove empty, deleted and removed comments
          if comment in {'', '[deleted]', '[removed]'}:
            continue

          comments_text += '\n' + '****TC ' + post_id + '\n' + str(comment) + '\n' + '****ETC' + '\n'
        post_text = post_text + '\n' + comments_text
      except Exception:
        post_text = ''

      # Append to respective post_id in the dictionary
      if post_id not in self.post_texts_with_comments:
          self.post_texts_with_comments[post_id] = post_text
      else:
        self.post_texts_with_comments[post_id] = self.post_texts_with_comments[post_id] + '\n' + post_text

  def preprocess_text(self, df):
    self.df = df
    unique_post_ids = self.df['id'].unique()

    # Apply multiprocessing to each post_id
    pool = Pool(processes = 8)
    for index, _ in enumerate(pool.imap(self._multi_preprocess, unique_post_ids)):
      print(index, len(unique_post_ids), index * 100/ len(unique_post_ids))
    pool.close()

  def save_text(self, file_name = 'train.txt'):
    print("SAVING TEXT")
    texts = list(self.post_texts_with_comments.values())
    train_text = '\n'.join(texts)
    text_file = open("train.txt", "w")
    text_file.write(train_text)
    text_file.close()
