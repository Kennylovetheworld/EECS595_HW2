import numpy as np
import pickle
import pickle, os, sys
import torch 
from torch import nn
import torch.optim as optim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

GLOVE_FILE = 'glove.6B/glove.6B.300d.txt'
VEC_LENGTH = 300
PADDING_LENGTH = 30
TRAINING_FILE = 'wsj1-18.training'
VALIDATION_FILE = 'wsj19-21.truth'
NEG_Path = "dataset/training/neg/"
NEG_Files= os.listdir(NEG_Path)
POS_Path = "dataset/training/pos/"
POS_Files= os.listdir(POS_Path)
ps = PorterStemmer()

class data_process():
    def __init__(self, glove_model):
        self.glove_model = glove_model

    def data_embedding(self, neg_s, pos_s):
        outname = 'log/data.pkl'
        if os.path.isfile(outname):
            data = pickle.load(open(outname,'rb'))
        else:
            label = []
            embedded_sents = [] 
            for sent in neg_s:
                label.append(0)
                sent = word_tokenize(sent)
                sent_vec = np.zeros((PADDING_LENGTH, VEC_LENGTH))
                for i, word in enumerate(sent):
                    if i == PADDING_LENGTH:
                        break
                    if word in self.glove_model:
                        vec = self.glove_model[word]
                    elif ps.stem(word) in self.glove_model:
                        vec = self.glove_model[ps.stem(word)]
                    else:
                        vec = np.zeros(300)
                    sent_vec[i] = vec
                embedded_sents.append(sent_vec)
            for sent in pos_s:
                label.append(1)
                sent = word_tokenize(sent)
                sent_vec = np.zeros((PADDING_LENGTH, VEC_LENGTH))
                for i, word in enumerate(sent):
                    if i == PADDING_LENGTH:
                        break
                    if word in self.glove_model:
                        vec = self.glove_model[word]
                    elif ps.stem(word) in self.glove_model:
                        vec = self.glove_model[ps.stem(word)]
                    else:
                        vec = np.zeros(300)
                    sent_vec[i] = vec
                embedded_sents.append(sent_vec)
            label = np.array(label)
            embedded_sents = np.array(embedded_sents)
            data = {}
            data['label'] = label
            data['features'] = embedded_sents
            pickle.dump(data, open(outname,'wb'))
        return data


    def sentence_tokenize(self):
        neg_outname = 'log/neg_sents.pkl'
        pos_outname = 'log/pos_sents.pkl'
        # Tokenize NEG data
        if os.path.isfile(neg_outname):
            neg_s = pickle.load(open(neg_outname,'rb'))
        else:
            neg_s = []
            for file_name in NEG_Files:
                with open(NEG_Path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    sents = sent_tokenize(data)
                    neg_s.extend(sents)
            pickle.dump(neg_s, open(neg_outname,'wb'))
        # Tokenize POS data
        if os.path.isfile(pos_outname):
            pos_s = pickle.load(open(pos_outname,'rb'))
        else:
            pos_s = []
            for file_name in POS_Files:
                with open(POS_Path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    sents = sent_tokenize(data)
                    pos_s.extend(sents)
            pickle.dump(pos_s, open(pos_outname,'wb'))
        return neg_s, pos_s
      
def loadGloveModel(gloveFile):
    outname = 'log/' + gloveFile + '.pkl'
    if os.path.isfile(outname):
        model = pickle.load(open(outname,'rb'))
    else:
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        pickle.dump(model, open(outname,'wb'))
    return model

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size = 300, hidden_size = 45, bidirectional=True)
        self.linear = nn.Linear(90, 45)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, seq, hidden=None):
        output, _ = self.rnn(seq)
        output = self.softmax(self.linear(output))
        return output

model = loadGloveModel(GLOVE_FILE)
dp = data_process(model)
neg_s, pos_s = dp.sentence_tokenize()
data = dp.data_embedding(neg_s, pos_s)
labels = data['label']
inputs = data['features']

RNN = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')