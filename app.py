import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import sys
import gensim
import getopt
import pandas
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, metrics, svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier

TRAIN_FILE = "train.csv"
INPUT_FILE = "predict.csv"
OUTPUT_FILE = "results.csv"
W2V_FILE = "glove.6B.50d.txt"
FOLDS = 10

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
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

def load_data(data_path):
    data_file = pandas.read_csv(data_path)
    x_text = data_file.Text.tolist()
    return [s.strip() for s in x_text], [s.strip().split() for s in x_text]

def preprocess_w2v(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        X_train_value, X_test_value = X[train_index], X[test_index]
        y_train_value, y_test_value = y[train_index], y[test_index]
        X_train.append(X_train_value)
        X_test.append(X_test_value)
        y_train.append(y_train_value)
        y_test.append(y_test_value)
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
    preds = []
    mean_tpr = 0.0
    fpr, tpr, roc_auc = {}, {}, {}
    mean_fpr = np.linspace(0, 1, 100)
    acc_tot = precision_tot = recall_tot = f1_tot = support_tot = 0
    headers = 'fl,precision,recall,f1,support,acc,TN, FP, FN, TP'
    print(headers)
    for fl in range(FOLDS):
        clf.fit(X_train[fl], y_train[fl])
        pred = clf.predict(X_test[fl])
        acc = metrics.accuracy_score(y_test[fl], pred)
        conf_matrix = metrics.confusion_matrix(y_test[fl], pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_test[fl], pred, average="weighted")
        support = conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[1][0]+conf_matrix[1][1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test[fl], pred)
        roc_auc = metrics.auc(fpr, tpr)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (fl, precision, recall, f1, support, acc, conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]))
        acc_tot += acc
        precision_tot += precision
        recall_tot += recall
        f1_tot += f1
        support_tot += support
    mean_tpr /= FOLDS
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    acc_tot /= FOLDS
    precision_tot /= FOLDS
    recall_tot /= FOLDS
    f1_tot /= FOLDS
    support_tot /= FOLDS
    print("Total accuracy: "+str(acc_tot))
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr, 
        'mean_auc': mean_auc
    }

def plot(title, records):
    lw = 2
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    for record in records:
        desc = '%s (area = %0.2f)' % (record['label'], record['mean_auc'])
        plt.plot(record['mean_fpr'], record['mean_tpr'], linestyle='--', label=desc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def benchmark(input_file):
    X, y = load_data_and_labels(input_file)
    clfs = [ExtraTreesClassifier(n_estimators=200),
            svm.SVC(C=1, kernel='linear'),
            KNeighborsClassifier(n_neighbors=3),
            DecisionTreeClassifier(random_state=0),
            MLPClassifier(alpha=1),
            GaussianNB()]
    names = ['ExtraTrees', 'SVM', 'KNN', 'DecisionTree', 'NeuralNetwork', 'NaiveBayes']
    print("WORD2VEC")
    with open(W2V_FILE, "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    X_train, X_test, y_train, y_test = preprocess_w2v(X,y)
    cont = 0
    sections = ['Word2Vec', 'BagOfWords']
    results = {}
    section = sections[cont]
    print(section)
    results[section] = {}
    for i in range(len(names)):
        print(section + " " + names[i])
        clf = Pipeline([
            (section + " vectorizer", MeanEmbeddingVectorizer(w2v)),
            ("classifier", clfs[i])])
        results[section][names[i]] = train(clf, X_train, X_test, y_train, y_test)
    cont += 1
    section = sections[cont]
    print(section)
    results[section] = {}
    X_train, X_test, y_train, y_test = preprocess_bag(X,y)
    for i in range(len(names)):
        print(section + " " + names[i])
        clf = clfs[i]
        results[section][names[i]] = train(clf, X_train, X_test, y_train, y_test)
    for name in names:
        records = []
        for section in sections:
            values = results[section][name]
            values['label'] = section
            records.append(values)
        plot('Comparison using ' + name, records)

def predict(params):
    clf = ExtraTreesClassifier(n_estimators=200)
    with open(params["w2v_file"], "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    clf_w2v = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("classifier", clf)])
    X, y = load_data_and_labels(params["train_file"])
    col_y, arr_X, arr_y = {}, [], []
    for i in range(len(y)):
        if y[i] not in col_y:
            col_y[y[i]] = []
        col_y[y[i]].append(X[i])
    for field in col_y:
        arr_X.append(col_y[field])
        arr_y.append(field)
    clf_w2v.fit(arr_X, arr_y)
    raw_data = {}
    raw_data['Text'], tokenized = load_data(params["input_file"])
    raw_data['Class'] = clf_w2v.predict(tokenized)
    data_frame = pandas.DataFrame(raw_data)
    data_frame.to_csv(params["output_file"])
    print(data_frame)
    return raw_data

def run(params):
    if params["bench"]:
        benchmark(params)
    else:
        predict(params)

def main(argv):
    params = {
        "bench": False,
        "train_file": TRAIN_FILE,
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,
        "w2v_file": W2V_FILE
    }
    opts, args = getopt.getopt(argv, "t:i:o:w:b")
    if opts:
        for o, a in opts:
            if o in ("-b"):
                params["bench"] = True
            elif o in ("-t"):
                params["train_file"] = a
            elif o in ("-i"):
                params["input_file"] = a
            elif o in ("-o"):
                params["output_file"] = a
            elif o in ("-w"):
                params["w2v_file"] = a
    run(params)

if __name__ == "__main__":
    main(sys.argv[1:])


# page
# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

"""
clfs = [ExtraTreesClassifier(n_estimators=200),
        svm.SVC(C=1, kernel='linear'),
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(random_state=0),
        MLPClassifier(alpha=1),
        GaussianNB()]
names = ['ExtraTrees', 'SVM', 'KNN', 'DecisionTree', 'NeuralNetwork', 'NaiveBayes']
"""

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

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

# print(metrics.classification_report(y_test[fl], pred))
# print("Acc:"+str(acc))
# print(metrics.confusion_matrix(y_test[fl], pred))
# mean_tpr[0] = 0.0
# plt.plot(fpr, tpr, lw=lw, label='ROC fold %d (area = %0.2f)' % (fl, roc_auc))

# print("Fold "+str(fl))

