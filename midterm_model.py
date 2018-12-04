import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import numpy as np
#read in data
data = pd.read_csv("/Users/timklabjan/Downloads/Task3_content_topic_prediction/Topic_prediction_data_train.txt",sep='\t',header=None)
data.columns = ['text','uid','type','timestamp']
labels = pd.read_csv("/Users/timklabjan/Downloads/Task3_content_topic_prediction/Topic_prediction_labels_train.txt",sep='\t',header=None)
labels.columns = ['label']
data['label'] = labels['label']
del(labels)

#drop null values
data = data[(data['text'].notnull())]

#convert text to bag of words
corpus = data['text'].values
v = CountVectorizer()
bow = pd.DataFrame(v.fit_transform(corpus).todense())
bow['label'] = data['label'].astype(int).astype(str)
bow = bow[bow['label'].notnull()]
inputs = bow[bow.columns.tolist()[:-1]]
labels = bow['label']
del(bow)

#k fold cross validation
kf = KFold(n_splits=3,random_state=387)
kf.get_n_splits(inputs)

accuracies = []

for train_index, test_index in kf.split(inputs):
    print("training model")
    X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #train model
    model = LogisticRegression(random_state=387)
    model = model.fit(X_train,y_train)
    # get predictions and accuracy
    pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))
print(accuracies)
print(sum(accuracies)/3.0)
print(accuracy_score(y_test,np.random.choice(['0','1','2','3','4','5','6','7','8','9'],size=len(y_test),replace=True)))
