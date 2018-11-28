import keras
import numpy as np
import pandas as pd
import pickle
import sys
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils import np_utils

data_path=  sys.argv[1]
#data_path="Topic_prediction_data_train.txt"
data = pd.read_csv(data_path,sep='\t',header=None)
data.columns = ['text','uid','type','timestamp']

#labels = pd.read_csv("Topic_prediction_labels_train.txt",sep='\t',header=None)
#labels.columns = ['label']
#data['label'] = labels['label']





data = data[(data['text'].notnull())]
data=data.reset_index(drop=True)


#labels = np.array(data["label"])
#labels = np_utils.to_categorical(labels, num_classes=10)

alltext = [i.split(' ') for i in data["text"]]


with open('word_dictionary2.pickle', 'rb') as reader:
    w_dic= pickle.load(reader)


def convert_to_int(words):
    converted=[]
    for w in words:
        if w in w_dic:
            converted.append(w_dic[w])
    return converted


data_conv =[]
for item in alltext:
    data_conv.append(convert_to_int(item))

data_conv=np.array(data_conv)

# pad
data_conv_trunc= sequence.pad_sequences(data_conv,padding='post')
x_Test= data_conv_trunc[:,:350]
#x_Test=x_Test[1:100]
#print(x_Test.shape)


#load frozen model and make predictions
model_loaded = load_model('my_model.h5')
predictions = model_loaded.predict_classes(x_Test, verbose=1)
print(str(predictions))
#evp=model_loaded.evaluate(x_Test, labels[70000:72000], verbose=1)
#print(evp)

#save predicted output to file.
with open('rnn_output.txt','w') as f:
    for item in predictions:
        f.write("%s\n" % item)