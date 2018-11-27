import pickle
import sys
import pandas as pd
import numpy as np

#load in saved model and term frequency vectorizer
model = pickle.load(open("ensemble_model.p","rb"))
vectorizer = pickle.load(open("vectorizer.p","rb"))

#read in dataset
file_name = sys.argv[1]
data = pd.read_csv(file_name,sep='\t',header=None)
data.columns = ['text','uid','type','timestamp']
data = data[(data['text'].notnull())]

feats = np.array(vectorizer.transform(data["text"]).todense())
predictions = model.predict(feats)
with open('ensemble_output.txt','w') as f:
    for item in predictions:
        f.write("%s\n" % item)