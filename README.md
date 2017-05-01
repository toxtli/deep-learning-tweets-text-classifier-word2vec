# deep-learning-tweets-classifier

This tool uses Word2Vec and Bag of Words combined with Neural Networks, SVN, KNN, Naive Bayes, Decision Trees and ExtraTrees to classify tweets according to an annotated file.

Run this in command line:

> wget http://nlp.stanford.edu/data/glove.6B.zip

> unzip glove.6B.zip

> sudo pip install -r requirements.txt

Place your data.csv in the directory. It only requires the column "Text" where the tweets text is placed and the column "Class" than contains the annotated categories.

Run the python script:

> python app.py

The tool will print the results in the terminal.
