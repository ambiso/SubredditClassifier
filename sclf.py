#!/usr/bin/env python3

import numpy as np
import bisect
import praw
from tqdm import tqdm
import pickle
from random import shuffle

def map_idx(idxs, l):
    return [l[idx] for idx in idxs]

def shuf_lists(lists):
    length = len(lists[0])
    for l in lists:
        assert(length == len(l))
    idxs = list(range(length))
    shuffle(idxs)
    return [map_idx(idxs, l) for l in lists]

class RedditData:
    def __init__(self, data, target, target_names):
        assert len(data) == len(target)
        self.data = data
        self.target = target
        self.target_names = target_names

    def test_split(self, percent):
        [data,target] = shuf_lists([self.data, self.target])
        length = len(data)
        assert length == len(target)
        split_idx = ((percent * length) // 100) + 1
        return [RedditData(data[:split_idx], target[:split_idx], self.target_names),
                RedditData(data[split_idx:], target[split_idx:], self.target_names)]

def load_reddit_data(subreddits, num_posts):
    r = praw.Reddit(user_agent='RedditData', client_id="_wIfutjWp9ldaQ", client_secret="pAj3opavUHR1PYx4n-_fd9ZPn1E")
    data = []
    target = []
    for i, sr in tqdm(enumerate(subreddits), total=len(subreddits)):
        for post in r.subreddit(sr).hot(limit=num_posts):
            data.append(post.title + post.selftext)
            target.append(i)
    return RedditData(data, target, subreddits)

def main():
    pickle_file = "reddit.pickle"
    try:
        with open(pickle_file, "rb") as f:
            print("Loading data from", pickle_file)
            reddit = pickle.load(f)
    except FileNotFoundError:
        print("Receiving data")
        reddit = load_reddit_data(["minecraft","askscience","computerscience","mildlyinteresting", "nottheonion"], 10000)
        print("Saving to", pickle_file)
        with open(pickle_file, "wb") as f:
            pickle.dump(reddit, f)

    print("Got:", len(reddit.data))


    print("Learning..")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    #from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    text_clf = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        #('clf', MultinomialNB()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
    ])
    learning_data, test_data = reddit.test_split(80)
    print("Learning data:", len(learning_data.data))
    print("Test data:", len(test_data.data))
    text_clf = text_clf.fit(learning_data.data, learning_data.target)

    print("Evaluating")
    true = test_data.target
    pred = text_clf.predict(test_data.data)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print("Accuracy:           ", accuracy_score(true, pred))
    print("Normalized Accuracy:", accuracy_score(true, pred, normalize=True))
    print("Micro Precision:    ", precision_score(true, pred, average='micro'))
    print("Macro Precision:    ", precision_score(true, pred, average='macro'))
    print("Micro Recall:       ", recall_score(true, pred, average='micro'))
    print("Macro Recall:       ", recall_score(true, pred, average='macro'))

    while True:
        text = input("Input: ")
        docs_new = [text]
        predicted = text_clf.predict(docs_new)
        for doc, category in zip(docs_new, predicted):
            print('%r => %s' % (doc, reddit.target_names[category]))

if __name__ == "__main__":
    main()

