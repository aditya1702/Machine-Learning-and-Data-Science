class DatasetLoader:
    def __init__(self, data, subset = None, holdout = 0.1):
        self.subredditToLabel = {'funny': 0,
                                 'soccer': 1,
                                 'teenagers': 2,
                                 'machinelearning': 3,
                                 'gameofthrones': 4,
                                 'Showerthoughts': 5,
                                 'unpopularopinion': 6,
                                 'politics': 7,
                                 'worldnews': 8}
        self.data = data
        self.subset = subset
        self.holdout = holdout

    def transform(self):
        try:
            self.data["subreddit"] = self.data["subreddit"].apply(lambda x: self.subredditToLabel[x])
        except:
            pass

        if self.subset:
            self.data = (self.data.groupby('subreddit', group_keys = False).apply(lambda x: x.sample(min(len(x), self.subset)))).reset_index()
            self.data = self.data.sample(frac = 1).reset_index(drop = True)

    def prepare_holdout(self):
        holdout_size = int(len(self.data) * self.holdout)

        self.sentences = self.data.title.values
        self.test_sentences = self.sentences[-holdout_size:]
        self.sentences = self.sentences[:-holdout_size]
        self.labels = self.data.subreddit.values
        self.test_labels = self.labels[-holdout_size:]
        self.labels = self.labels[:-holdout_size]
