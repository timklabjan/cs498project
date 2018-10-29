import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

#train/test data split
x_train, x_test, y_train, y_test = train_test_split(bow[bow.columns.tolist()[:-1]],bow['label'],test_size=0.25,random_state=387)

#train model
model = LogisticRegression()
model = model.fit(x_train,y_train)

#get predictions and accuracy
pred = model.predict(x_test)
print(accuracy_score(y_test,pred))
print(accuracy_score(y_test,np.random.choice(['0','1','2','3','4','5','6','7','8','9'],size=len(y_test),replace=True)))
