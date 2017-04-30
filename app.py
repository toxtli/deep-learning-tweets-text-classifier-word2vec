import gensim
import pandas
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, metrics, svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

FOLDS = 10
W2V_FILE = "glove.6B.300d.txt"

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def load_data_and_labels(data_path):
    data_file = pandas.read_csv(data_path)
    x_text = data_file.Text.tolist()
    x_text = [s.strip() for s in x_text]
    labels = data_file.Class.tolist()
    return x_text,labels

def preprocess_w2v(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def preprocess_bag(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        features_train, features_test = X[train_index], X[test_index]
        t_labels_train, t_labels_test = y[train_index], y[test_index]
        vectorizer = TfidfVectorizer(stop_words='english')
        features_train = vectorizer.fit_transform(features_train)
        features_test  = vectorizer.transform(features_test)
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(features_train, t_labels_train)
        X_train.append(selector.transform(features_train).toarray())
        X_test.append(selector.transform(features_test).toarray())
        y_train.append(t_labels_train)
        y_test.append(t_labels_test)
    return X_train, X_test, y_train, y_test

def train(clf, X_train, X_test, y_train, y_test):
    acc_tot = 0
    for fl in range(FOLDS):
        clf.fit(X_train[fl], y_train[fl])
        pred = clf.predict(X_test[fl])
        acc = metrics.accuracy_score(y_test[fl], pred)
        print("Fold "+str(fl)+" acc:"+str(acc))
        print(metrics.classification_report(y_test[fl], pred))
        acc_tot += acc    
    acc_tot /= FOLDS
    print("Total accuracy: "+str(acc_tot))

X, y = load_data_and_labels("data.csv")
clfs = [ExtraTreesClassifier(n_estimators=200),
        svm.SVC(C=1, kernel='linear'),
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(random_state=0),
        MLPClassifier(alpha=1),
        GaussianNB()]
names = ['ExtraTrees', 'SVM', 'KNN', 'Decision Tree', 'Neural Network', 'Naive Bayes']

print("WORD2VEC")
with open(W2V_FILE, "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
X_train, X_test, y_train, y_test = preprocess_w2v(X,y)

for i in range(len(names)):
    print("Word2Vec " + names[i])
    clf = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("classifier", clfs[i])])
    train(clf, X_train, X_test, y_train, y_test)

print("BAG OF WORDS")
X_train, X_test, y_train, y_test = preprocess_bag(X,y)
for i in range(len(names)):
    print("BagOfWords " + names[i])
    clf = ExtraTreesClassifier(n_estimators=200)
    train(clf, X_train, X_test, y_train, y_test)

"""
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
print("Word2Vec ExtraTrees TfidfEmbedding")
train(etree_w2v_tfidf, X_train, X_test, y_train, y_test)
"""

# data_path = 'data.csv'
# data_file = pandas.read_csv(data_path)
# x_text = data_file.Text.tolist()
# x_text = [s.strip() for s in x_text]
# model = gensim.models.Word2Vec(x_text, size=100)
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))

# X = [['Berlin', 'London'],
#      ['cow', 'cat'],
#      ['pink', 'yellow'],
#      ['cold', 'hot']]

# X = [['I', 'love', 'like', 'morphine'],
#      ['I', 'use', 'consume', 'morphine']]
# y = ['non addict', 'addict']

# never before seen words!!!
# test_X = [['dog'], ['green'], ['Madrid']]
# test_X = [['I', 'consume', 'morphine']]
# print(etree_w2v.predict(test_X))

# acc = metrics.accuracy_score(y_test,etree_w2v.predict(X_test))
# acc_tfidf = metrics.accuracy_score(y_test,etree_w2v_tfidf.predict(X_test))
# print("model acc:"+str(acc))
# print("model acc_tfidf:"+str(acc_tfidf))