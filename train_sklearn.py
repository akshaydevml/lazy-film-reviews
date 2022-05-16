import argparse
import os
import pickle
import pprint
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model',
    help='Model to be used in the Pipeline',
    choices=['LinearSVC', 'SGDClassifier', 'GradientBoostingClassifier'],
    default='LinearSVC')
parser.add_argument('--nfold',
                    help='Select number of folds for cross-validation',
                    type=int,
                    default=5)
parser.add_argument('--seed_val', help='Seed value', type=int, default=42)
parser.add_argument('csvfile', help='Input csv file')

args = parser.parse_args()
seed_all(args.seed_val)
train_df = pd.read_csv(args.csvfile)

encoder = preprocessing.LabelEncoder()
train_df['sentiment'] = encoder.fit_transform(train_df['sentiment'])

if args.model == "SGDClassifier":
    model = SGDClassifier(random_state=args.seed_val, tol=1e-05)
elif args.model == "LinearSVC":
    model = LinearSVC(random_state=args.seed_val, tol=1e-05)
elif args.model == "GradientBoostingClassifier":
    model = GradientBoostingClassifier(random_state=args.seed_val, tol=1e-05)

cls_pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
                         ('clf', model)])
kf = KFold(n_splits=args.nfold, random_state=args.seed_val, shuffle=True)

results = cross_validate(
    cls_pipeline,
    train_df['review'],
    train_df['sentiment'],
    scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    cv=kf,
    return_estimator=True,
    return_train_score=True)

print('Cross-Validation Results')
pprint.pprint(results)
best_fold = np.argmax(results['test_roc_auc'])
filename = 'movie_sentiment.pkl'
model = results['estimator'][best_fold]
print((f'Saving model at fold {best_fold} with'
       f' roc auc value {np.max(results["test_roc_auc"])}'))
pickle.dump(model, open(filename, 'wb'))
