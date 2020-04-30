import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def data_loader_tfidf(file):
    # Function to read and process train data file with TFIDF vectorizer
    train = pd.read_csv(file)
    # Remove na comments
    train = train.dropna(subset = ['comment'])
    train = train.iloc[:, :-3].drop('author', axis = 1)
    train_x = train.drop('label', axis = 1)
    train_y = train[['label']]
    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)
    train_tfidf_1 = tfidf_vectorizer.fit_transform(train_x['comment'])
    # print(train_tfidf_1.shape)
    # tfidf_vectorizer_new = TfidfVectorizer(ngram_range=(1, 1), max_features = 100)
    # train_tfidf_2 = tfidf_vectorizer_new.fit_transform(train_x['subreddit'])
    # print(train_tfidf_2.shape)
    # scores = train_x.iloc[:,-3:]
    # scaler = StandardScaler()
    # scaled_scores = scaler.fit_transform(scores)
    # print(scaled_scores.shape)
    # train_x_new = hstack([train_tfidf_1, train_tfidf_2, scaled_scores])
    # print(train_x_new.shape)
    train_x_new = hstack([train_tfidf_1])
    x_train, x_test, y_train, y_test = train_test_split(train_x_new, train_y, test_size = 0.15)
    
    return x_train, x_test, y_train, y_test


def data_loader_count(file):
    # Function to read and process train data file with count vectorizer
    train = pd.read_csv(file)
    # Remove na comments
    train = train.dropna(subset = ['comment'])
    train = train.iloc[:, :-3].drop('author', axis = 1)
    train_x = train.drop('label', axis = 1)
    train_y = train[['label']]
    count_vectorizer = CountVectorizer(ngram_range = (1, 2), max_features = 1000)
    train_count_1 = count_vectorizer.fit_transform(train_x['comment'])
    # print(train_count_1.shape)
    # count_vectorizer_new = CountVectorizer(ngram_range=(1, 1), max_features = 100)
    # train_count_2 = count_vectorizer_new.fit_transform(train_x['subreddit'])
    # print(train_count_2.shape)
    # scores = train_x.iloc[:,-3:]
    # scaler = StandardScaler()
    # scaled_scores = scaler.fit_transform(scores)
    # print(scaled_scores.shape)
    # train_x_new = hstack([train_count_1, train_count_2, scaled_scores])
    # print(train_x_new.shape)
    train_x_new = hstack([train_count_1])
    x_train, x_test, y_train, y_test = train_test_split(train_x_new, train_y, test_size = 0.15)
    
    return x_train, x_test, y_train, y_test


def model(x_train, x_test, y_train, y_test):
    # Function to train and test the Support Vector Machine model
    clf = LinearSVC(random_state = 0, tol = 1e-5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Train model accuracy: ", clf.score(x_train, y_train))
    print("Test model accuracy: ", clf.score(x_test, y_test))
    print("F1 score:", f1_score(y_test, y_pred, average = "macro"))
    print("Precision: ", precision_score(y_test, y_pred, average = "macro"))
    print("Recall: ", recall_score(y_test, y_pred, average = "macro")) 
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ")
    print(conf_matrix)
    
    
def main():
    file_name = 'train-balanced-sarcasm.csv'
    # without meta-data - only comments
    # uncomment the comments to include meta-data
    print("Support Vector Machine for TF-IDF vectorization: ")
    x_train, x_test, y_train, y_test = data_loader_tfidf(file_name)
    model(x_train, x_test, y_train, y_test)
    print("\n\nSupport Vector Machine for Count vectorization: ")
    x_train, x_test, y_train, y_test = data_loader_count(file_name)
    model(x_train, x_test, y_train, y_test)    