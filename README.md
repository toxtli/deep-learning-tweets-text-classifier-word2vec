# deep-learning-tweets-text-classifier-word2vec

This tool uses Word2Vec and Bag of Words combined with Neural Networks, SVN, KNN, Naive Bayes, Decision Trees and ExtraTrees to classify tweets according to an annotated file. This classifier was used to classify tweets from Twitter.

Run this in command line:

> wget http://nlp.stanford.edu/data/glove.6B.zip

> unzip glove.6B.zip

> sudo pip install -r requirements.txt

Demo files are in place in order to have a clear idea about how this works.

Run the python script:

> python app.py

or

> python app.py -t train.csv -i predict.csv -o results.csv -w glove.6B.50d.txt

The tool will print the results in the terminal and store in the results.csv file.

This classifier is multilingual, for instance if you want to run it in spanish you can use this word vector file http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2 instead of the glove file.
