import os
import json
import numpy as np
import nltk
import random
import keras.preprocessing.text as kpt
import keras.preprocessing.sequence as kps
import keras
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stopWords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
maxLength = 1000
maxVocab = 20000
def addToLabels(texts,labels,tokenizedText,jsonData):
    totalviolated = getViolated(jsonData)
    texts.append(tokenizedText)
    if totalviolated > 0:
        labels.append(1)
    else:
        labels.append(0)
def getViolated(jsonData):
    article = jsonData['VIOLATED_ARTICLES']
    paragraph = jsonData['VIOLATED_PARAGRAPHS']
    bulletpoint = jsonData['VIOLATED_BULLETPOINTS']
    totalviolated = len(article) + len(paragraph) + len(bulletpoint)
    return totalviolated
def preProcessing(value):
    stringList="".join(value)
    list = nltk.word_tokenize(stringList)
    list = [word.lower() for word in list if word.isalpha()]
    #list = [w for w in list if not w in stopWords]
    list = ([lemmatizer.lemmatize(x) for x in list])
    newList=""
    newL=[]
    for x in list:
        newList+=x
        newList+=" "
    if(len(newList)==0):
      newL.append(value)
    else:
      newL.append(newList)
    return  newList
trainPath = "/content/drive/MyDrive/EN_train"
validationPath="/content/drive/MyDrive/EN_dev"
testPath = "/content/drive/MyDrive/EN_test"
#trainPath = r"C:\Users\user\PycharmProjects\pythonProject\EN_train1"
#testPath = r"C:\Users\user\PycharmProjects\pythonProject\EN_test1"

jsonFilesTrain = [x for x in os.listdir(trainPath) if x.endswith("json")]
jsonFilesValidation = [x for x in os.listdir(validationPath) if x.endswith("json")]
jsonFilesTest = [x for x in os.listdir(testPath) if x.endswith("json")]
traintexts = []
trainlabels = []
validationtexts = []
validationlabels = []
testtexts = []
testlabels = []
tryCounter=0
fileCounter=0
tokenizedLen = 0
for json_file in jsonFilesTrain:
    datasetPath = os.path.join(trainPath, json_file)
    with open(datasetPath, "r") as f:
        jsonData = json.load(f)
        value = jsonData['TEXT']

        flattenString=""
        for i in range(0,len(value)):
            text = []
            text=preProcessing(value[i])
            addToLabels(traintexts, trainlabels, text, jsonData)
for json_file in jsonFilesValidation:
    datasetPath = os.path.join(validationPath, json_file)
    with open(datasetPath, "r") as f:
        jsonData = json.load(f)
        value = jsonData['TEXT']
        text = []
        flattenString = ""
        for i in range(0,len(value)):
            flattenString+=value[i]
        addToLabels(validationtexts,validationlabels,flattenString,jsonData)

for json_file in jsonFilesTest:
    datasetPath = os.path.join(testPath, json_file)
    with open(datasetPath, "r") as f:
        jsonData = json.load(f)
        value = jsonData['TEXT']
        text = []
        flattenString = ""
        for i in range(0,len(value)):
            flattenString+=value[i]
        addToLabels(testtexts,testlabels,flattenString,jsonData)
maxVocab=5000
tokenizer = kpt.Tokenizer(maxVocab)

tokenizer.fit_on_texts(traintexts)
trainTexts = traintexts
trainLabels = trainlabels
validationTexts=validationtexts
validationLabels=validationlabels
testTexts = testtexts
testLabels = testlabels

sequences = tokenizer.texts_to_sequences(trainTexts)
validationSequences=tokenizer.texts_to_sequences(validationTexts)
testSequences = tokenizer.texts_to_sequences(testTexts)
trainData = kps.pad_sequences(sequences,512, padding="pre", truncating="pre")
validationData = kps.pad_sequences(validationSequences,512, padding="pre", truncating="pre")
testData = kps.pad_sequences(testSequences,512, padding="pre", truncating="pre")

messages_train = np.asarray(trainData)
labels_train =np.asarray(trainLabels)
messages_validation = np.asarray(validationData)
labels_validation =np.asarray(validationLabels)
messages_test = np.asarray(testData)
labels_test = np.asarray(testLabels)
embedding_mat_colums = 32
from keras.constraints import maxnorm
import keras.regularizers as kr
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1000)
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=maxVocab, output_dim=embedding_mat_colums, input_length=512))
model.add(keras.layers.LSTM(units=128, dropout=0, recurrent_dropout=0, activation='tanh'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
initial_learning_rate=0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Precision(),keras.metrics.Recall()])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.FalsePositives(),keras.metrics.TruePositives(),keras.metrics.FalseNegatives(),keras.metrics.TrueNegatives()])
model.summary()
history=model.fit(messages_train, labels_train, epochs=8, batch_size=128, validation_data=(messages_validation,labels_validation))
model.save('lastModel.h5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
loss, accuracy, precision, recall , falsePositives , truePositives , falseNegatives, trueNegatives=model.evaluate(messages_test,labels_test,batch_size=128)