from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns

import pickle
import json
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Bidirectional, GRU
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder

main = tkinter.Tk()
main.title("CSKG4APT: A Cybersecurity Knowledge Graph for Advanced Persistent Threat Organization Attribution") #designing main screen
main.geometry("1000x650")

global dataset, X, Y, bilstm
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, labels, scaler, le1, le2

def loadDataset():
    global dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    labels = np.unique(dataset['apt'])
    label = dataset.groupby('apt').size()
    label.plot(kind="bar")
    plt.title("APT attacks found in Dataset")
    plt.show()

def knowledgeGraph():
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
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


def preprocessDataset():
    text.delete('1.0', END)
    global dataset, scaler, le1, le2, X, Y
    global X_train, X_test, y_train, y_test
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
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"80% training records : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% testing records  : "+str(X_test.shape[0]))

def calculateMetrics(predict, y_testData, algorithm):
    global labels
    global accuracy, precision, recall, fscore
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_testData, axis=1)
    precision = precision_score(y_test1, predict,average='macro') * 100
    recall = recall_score(y_test1, predict,average='macro') * 100
    fscore = f1_score(y_test1, predict,average='macro') * 100
    accuracy = accuracy_score(y_test1,predict)*100    
    text.insert(END,algorithm+' Accuracy  : '+str(accuracy)+"\n")
    text.insert(END,algorithm+' Precision : '+str(precision)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(recall)+"\n")
    text.insert(END,algorithm+' FMeasure  : '+str(fscore)+"\n\n")
        
    conf_matrix = confusion_matrix(y_test1, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runBILSTM():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, bilstm
    #now training BILSTM with GRU layer
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
        bilstm = load_model("model/bilstm.hdf5")
    predict = bilstm.predict(X_test)
    calculateMetrics(predict, y_test, "Propose Bi-LSTM with GRU Layers")


def predict():
    text.delete('1.0', END)
    global bilstm, le1, le2, scaler, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset['md5'] = pd.Series(le1.transform(dataset['md5'].astype(str)))
    dataset['apt'] = pd.Series(le2.transform(dataset['apt'].astype(str)))
    dataset.drop(['apt'], axis = 1,inplace=True)
    dataset = dataset.values
    X = scaler.transform(dataset)
    X = dataset[:,0:256]
    X = np.reshape(X, (X.shape[0], 16, 16))
    predict = bilstm.predict(X)
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        print(pred)
        text.insert(END,"Test Data = "+str(dataset[i])+" =====> Predicted APT Attack : "+labels[int(pred)]+"\n\n")

def graph():
    global accuracy, precision, recall, fscore
    height = [accuracy, precision, recall, fscore]
    bars = ('Accuracy','Precision','Recall','FScore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Comparison Matrics")
    plt.ylabel("Metric Values")
    plt.title("Propose Bi-LSTM with GRU Layer Comparison Graph")
    plt.show()


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='CSKG4APT: A Cybersecurity Knowledge Graph for Advanced Persistent Threat Organization Attribution', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload APT Attack Dataset", command=loadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

knowledgeButton = Button(main, text="Knowledge Graph from Dataset", command=knowledgeGraph)
knowledgeButton.place(x=330,y=100)
knowledgeButton.config(font=font1) 

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=620,y=100)
processButton.config(font=font1) 

biButton = Button(main, text="Run BI-LSTM with GRU Algorithm", command=runBILSTM)
biButton.place(x=10,y=150)
biButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=330,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Attack Detection from Test Data", command=predict)
predictButton.place(x=620,y=150)
predictButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=10,y=200)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
