import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Bidirectional, GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import os
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Dataset/APT_attack_dataset.csv")
dataset.fillna(0, inplace = True)
'''
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object':
        print(columns[i])


md = dataset['md5'].ravel()
apt = dataset['apt'].ravel()
dll = dataset['tapi16.dll'].ravel()

indices = np.arange(md.shape[0])
np.random.shuffle(indices) #shuffle the dataset
md = md[indices]
apt = apt[indices]
dll = dll[indices]

md = md[0:180]
apt = apt[0:180]
dll = dll[0:180]

graph = []
for i in range(len(md)):
    graph.append([md[i], apt[i], dll[i]])

knowledge_graph = pd.DataFrame(graph, columns=['Edge', 'Target', 'Source'])
print(knowledge_graph)
G = nx.from_pandas_edgelist(knowledge_graph, source='Source',target='Target',edge_attr='Edge', create_using = nx.MultiDiGraph())
plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()
'''
scaler = StandardScaler()

le1 = LabelEncoder()
le2 = LabelEncoder()
dataset['md5'] = pd.Series(le1.fit_transform(dataset['md5'].astype(str)))
dataset['apt'] = pd.Series(le2.fit_transform(dataset['apt'].astype(str)))

Y = dataset['apt'].ravel()
dataset.drop(['apt'], axis = 1,inplace=True)
X = dataset.values

X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices) #shuffle the dataset
X = X[indices]
Y = Y[indices]

X = X[:,0:256]
X = np.reshape(X, (X.shape[0], 16, 16))
print(X.shape)
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)

bilstm = Sequential()#defining deep learning sequential object
#adding bi-directional LSTM layer with 32 filters to filter given input X train data to select relevant features
bilstm.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
#adding dropout layer to remove irrelevant features
bilstm.add(Dropout(0.2))
#adding another layer
bilstm.add(Bidirectional(GRU(32)))
bilstm.add(Dropout(0.2))
#defining output layer for prediction
bilstm.add(Dense(y_train.shape[1], activation='softmax'))
#compile BI-LSTM model
bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#start training model on train data and perform validation on test data
if os.path.exists("model/bilstm.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/bilstm.hdf5', verbose = 1, save_best_only = True)
    hist = bilstm.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    bilstm.load_weights("model/bilstm.hdf5")

predict = bilstm.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, predict)
print(acc)


