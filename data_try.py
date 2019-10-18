import numpy as np
import pickle
import pickle, os, sys
import torch 
from torch import nn
import torch.optim as optim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
# from torchsummary import summary
# from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOVE_FILE = 'glove.6B/glove.6B.300d.txt'
VEC_LENGTH = 300
PADDING_LENGTH = 800
BATCH_SIZE = 20
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
        outname = 'log/data_doc_'+str(PADDING_LENGTH)+'.pkl'
        if os.path.isfile(outname):
            data = pickle.load(open(outname,'rb'))
        else:
            label = []
            embedded_sents = []
            lens = []
            for sent in neg_s:
                label.append([0,1])
                sent = word_tokenize(sent)
                lens.append(len(sent))
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
                label.append([0,1])
                sent = word_tokenize(sent)
                lens.append(len(sent))
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
            print(np.array(lens).std())
        return data

    def doc_tokenize(self):
        neg_outname = 'log/neg_docs.pkl'
        pos_outname = 'log/pos_docs.pkl'
        # Fetch NEG doc
        if os.path.isfile(neg_outname):
            neg_d = pickle.load(open(neg_outname,'rb'))
        else:
            neg_d = []
            for file_name in NEG_Files:
                with open(NEG_Path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    neg_d.append(data)
            pickle.dump(neg_d, open(neg_outname,'wb'))
        # Fetch POS doc
        if os.path.isfile(pos_outname):
            pos_d = pickle.load(open(pos_outname,'rb'))
        else:
            pos_d = []
            for file_name in POS_Files:
                with open(POS_Path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    pos_d.append(data)
            pickle.dump(pos_d, open(pos_outname,'wb'))
        return neg_d, pos_d

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
        self.rnn = nn.LSTM(input_size = 300, hidden_size = 50, bidirectional=True)
        self.linear = nn.Linear(100, 2)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, seq, hidden=None):
        output, _ = self.rnn(seq)
        output = self.softmax(self.linear(output))
        return output

class DAN(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(200, 200)
        self.linear2 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, seq, hidden=None):
        output, _ = self.rnn(seq)
        output = self.softmax(self.linear(output))
        return output

model = loadGloveModel(GLOVE_FILE)
dp = data_process(model)
neg_d, pos_d = dp.doc_tokenize()
data = dp.data_embedding(neg_d, pos_d)
labels = data['label']
features = data['features'].swapaxes(0,1)

net = RNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# train_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
# train_loader = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)

running_loss = 0.0
optimizer.zero_grad()
for iteration in range(1000):
    #training
    rand_indexs = list(np.random.random_integers(0, labels.shape[0] - 1, BATCH_SIZE))
    x = features[:, rand_indexs] 
    y = labels[rand_indexs]
    # forward + backward + optimize
    x = torch.tensor(x).float().reshape(-1, BATCH_SIZE, 300).to(DEVICE)
    y = torch.tensor(y).reshape(-1, 2).to(DEVICE)
    outputs = net(x)[-1,:,:]
    loss = criterion(y, outputs)
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    if (iteration+1) %10 == 0:
        print('[%d] loss: %.3f' %(iteration + 1, running_loss/20))
        running_loss = 0.0

print('Finished Training')