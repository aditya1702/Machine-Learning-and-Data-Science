import tensorflow as tf
import os
import gpt_2_simple as gpt2
import re
from random import shuffle


class GPT2Model:
  ModelsDirectory = './models/'
  SubredditMapping = {
    'politics': 't5_2cneq',
    'worldnews': 't5_2qh13',
    'learnmachinelearning': 't5_3cqa1',
    'unpopularopinion': 't5_2tk0s',
    'teenagers': 't5_2rjli',
    'MLQuestions': 't5_30rel',
    'soccer': 't5_2qi58',
    'gameofthrones': 't5_2rjz2',
    'Showerthoughts': 't5_2szyo',
    'deeplearning': 't5_2t5eh',
    'statistics': 't5_2qhfi',
    'funny': 't5_2fcsd'
  }

  Names = ['OnlyTone5',
           'Fabulous-Cold',
           'Original_Smile',
           'Downtown-Background',
           'Acceptable-Musician',
           'Think-Bicycle',
           'Southern_Appeal',
           'Prestigious-Artist',
           'Puzzleheaded-Amount',
           'IndependentExpert7',
           'Left-Mathematician',
           'LevelResponsibility9',
           'Spiritual_Customer',
           'RoutineGene8',
           'Technical-Line']

  def __init__(self, tf_sess = None, model_type = '124M', download_model = False):
    self.tf_sess = tf_sess
    if not tf_sess:
      self.tf_sess = gpt2.start_tf_sess()
    self.model_type = model_type

    # Download the model
    if download_model and not os.path.exists(self.ModelsDirectory + self.model_type):
      self.download_model()

  def fit(self,
          input_path,
          reset = True,
          overwrite = False,
          num_steps = 1000,
          batch_size = 1,
          print_every = 10,
          sample_every = 200,
          save_every = 300,
          restore_from = 'fresh',
          run_name = 'reddit_comment_generator'):
    if reset:
      tf.reset_default_graph()
      self.tf_sess = gpt2.start_tf_sess()

    if overwrite and restore_from != 'latest':
      restore_from = 'latest'

    # Finetuning the model on new data
    gpt2.finetune(self.tf_sess,
                  dataset = input_path,
                  batch_size = batch_size,
                  model_name = self.model_type,
                  steps = num_steps,
                  restore_from = restore_from,
                  run_name = run_name,
                  print_every = print_every,
                  sample_every = sample_every,
                  save_every = save_every)

  def generate_comments(self,
                        user_input,
                        bert_model_prediction,
                        length = 200,
                        temperature = 0.7,
                        num_samples = 2,
                        batch_size = 1,
                        top_k = 0,
                        top_p = 0,
                        run_name = 'reddit_comment_generator',
                        checkpoint_dir = './GPT2/checkpoint',
                        truncate_string = None):
    if not self.tf_sess:
      self.tf_sess = gpt2.start_tf_sess()

    # Generate samples
    subreddit_id = self.SubredditMapping[bert_model_prediction]
    prefix = '****S ' + subreddit_id + '\n' + user_input + '\n' + '****ES'

    comments = gpt2.generate(self.tf_sess,
                             length = length,
                             temperature = temperature,
                             prefix = prefix,
                             nsamples = num_samples,
                             batch_size = batch_size,
                             run_name = run_name,
                             top_k = top_k,
                             top_p = top_p,
                             return_as_list = True,
                             checkpoint_dir = checkpoint_dir,
                             truncate = truncate_string)

    index = 0
    shuffle(self.Names)
    ans = ''
    for text in comments:
        text = text.split('\n')
        L = len(text)

        i = 0
        while ('****TC' not in text[i]):
            text[i] = ''
            i += 1

        start = i
        while(i < L and '****S' not in text[i]):
            if '****TC' in text[i]:
              text[i] = '<strong>' + str(self.Names[index]) + '</strong>'
              index += 1
            elif '****ETC' in text[i]:
              text[i] = ''
            i += 1

        text = text[start:i]
        text = '\n'.join(text)
        if not ans:
            ans = text
        else:
            ans = ans + '\n\n' + text
    return ans

  def download_model(self):
    gpt2.download_gpt2(model_name = self.model_type)

  def load_pretrained_model(self, run_name = 'reddit_comment_generator', checkpoint_dir = './GPT2/checkpoint'):
    tf.reset_default_graph()
    self.tf_sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(self.tf_sess, run_name = run_name, checkpoint_dir = checkpoint_dir)
