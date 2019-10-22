import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import pickle, os, sys
import torch 
from torch import nn
import torch.optim as optim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from torchsummary import summary
from sklearn.utils import shuffle
# from torch.utils.data import DataLoader, TensorDataset

GLOVE_FILE = 'glove.6B/glove.6B.300d.txt'
VEC_LENGTH = 300
PADDING_LENGTH = 800
BATCH_SIZE = 50
TRAINING_FILE = 'wsj1-18.training'
VALIDATION_FILE = 'wsj19-21.truth'
NEG_Path = "dataset/training/neg/"
NEG_Files= os.listdir(NEG_Path)
POS_Path = "dataset/training/pos/"
POS_Files= os.listdir(POS_Path)

NEG_Path_val = "dataset/validation/neg/"
NEG_Files_val = os.listdir(NEG_Path_val)
POS_Path_val = "dataset/validation/pos/"
POS_Files_val = os.listdir(POS_Path_val)

NEG_Path_test = "dataset/testing/neg/"
NEG_Files_test = os.listdir(NEG_Path_test)
POS_Path_test = "dataset/testing/pos/"
POS_Files_test = os.listdir(POS_Path_test)

ps = PorterStemmer()
RNN_FILE = 'RNN.torch'

class data_process():
    def __init__(self, glove_model):
        self.glove_model = glove_model

    def data_embedding(self, neg_s, pos_s, validation = False, test = False):
        if validation is True and test is True:
            raise "unable to identify test or validation"
        if validation is True:
            outname = 'log/data_doc_validation_'+str(PADDING_LENGTH)+'.pkl'
        elif test is True:
            outname = 'log/data_doc_test_'+str(PADDING_LENGTH)+'.pkl'
        else:
            outname = 'log/data_doc_'+str(PADDING_LENGTH)+'.pkl'
        if os.path.isfile(outname):
            data = pickle.load(open(outname,'rb'))
        else:
            label = []
            embedded_sents = []
            lens = []
            for sent in neg_s:
                label.append(0)
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
                        vec = np.ones(300)
                    sent_vec[i] = vec
                embedded_sents.append(sent_vec)
            for sent in pos_s:
                label.append(1)
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
                        vec = np.ones(300)
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

    def doc_tokenize(self, validation = False, test = False):
        if validation is True and test is True:
            raise "unable to identify test or validation"
        if validation is True:
            neg_files = NEG_Files_val
            pos_files = POS_Files_val
            neg_path = NEG_Path_val
            pos_path = POS_Path_val
            neg_outname = 'log/neg_docs_val.pkl'
            pos_outname = 'log/pos_docs_val.pkl'
        elif test is True:
            neg_files = NEG_Files_test
            pos_files = POS_Files_test
            neg_path = NEG_Path_test
            pos_path = POS_Path_test
            neg_outname = 'log/neg_docs_test.pkl'
            pos_outname = 'log/pos_docs_test.pkl'
        else:
            neg_files = NEG_Files
            pos_files = POS_Files
            neg_path = NEG_Path
            pos_path = POS_Path
            neg_outname = 'log/neg_docs.pkl'
            pos_outname = 'log/pos_docs.pkl'
        # Fetch NEG doc
        if os.path.isfile(neg_outname):
            neg_d = pickle.load(open(neg_outname,'rb'))
        else:
            neg_d = []
            for file_name in neg_files:
                with open(neg_path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    neg_d.append(data)
            pickle.dump(neg_d, open(neg_outname,'wb'))
        # Fetch POS doc
        if os.path.isfile(pos_outname):
            pos_d = pickle.load(open(pos_outname,'rb'))
        else:
            pos_d = []
            for file_name in pos_files:
                with open(pos_path + file_name, encoding='windows-1252') as file:
                    data = file.read()
                    pos_d.append(data)
            pickle.dump(pos_d, open(pos_outname,'wb'))
        return neg_d, pos_d
      
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
        self.rnn = nn.LSTM(input_size=300, hidden_size=36, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(72, 2)
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.linear]:
            nn.init.kaiming_uniform_(layer.weight)

    def forward(self, seq, hidden=None):
        _, (hidden, _) = self.rnn(seq)
        hidden = torch.cat((hidden[0], hidden[1]), 1)
        output = self.linear(hidden)
        return output

model = loadGloveModel(GLOVE_FILE)
dp = data_process(model)
neg_d, pos_d = dp.doc_tokenize()
data = dp.data_embedding(neg_d, pos_d)
labels = data['label']
features = data['features']

neg_d, pos_d = dp.doc_tokenize(validation=True)
data = dp.data_embedding(neg_d, pos_d, validation=True)
labels_val = data['label']
features_val = data['features']

neg_d, pos_d = dp.doc_tokenize(test=True)
data = dp.data_embedding(neg_d, pos_d, test=True)
labels_test = data['label']
features_test = data['features']

net = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

#Training 
running_loss = 0.0
optimizer.zero_grad()
for epoch in range(10):
    if os.path.isfile(RNN_FILE):
            net = torch.load(RNN_FILE)
    running_loss = 0.0 # Reset the record of running_loss
    train_total_len = 0
    train_correct_number = 0
    features, labels = shuffle(features, labels)
    for iteration in range(20):
        #Choose a Batch
        # rand_indexs = list(np.random.random_integers(0, labels.shape[0] - 1, BATCH_SIZE)) 
        indexs = [index % features.shape[0] for index in range(iteration*50, (iteration + 1)*50)]
        x = features[indexs] 
        y = labels[indexs]
        # forward + backward + optimize
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        # updating statistics
        running_loss += loss.item()
        train_total_len += BATCH_SIZE
        y = y.detach().numpy()
        outputs = outputs.detach().numpy()
        outputs = np.array([np.where(item == np.amax(item)) for item in outputs])
        train_correct_number += sum([int(pred_y == target_y) for pred_y, target_y in zip(outputs,y)])
        
    print('---> Evaluating [%d] Epoch' %(epoch + 1))
    # Evaluate the accuracy on the training set
    accuracy = 100 * train_correct_number / train_total_len
    running_loss = running_loss/20
    # Print the result
    print('[%d] Training loss: %.6f, Training acc: %.3f%%.' %(epoch + 1, running_loss, accuracy))

    #Evaluate the accuracy on the validation set
    net.eval()
    x = torch.tensor(features_val).float()
    pred_val = net(x).detach().numpy()
    pred_val = np.array([np.where(item == np.amax(item)) for item in pred_val])
    true_val = labels_val
    val_correct_number = sum([int(pred_y == target_y) for pred_y, target_y in zip(pred_val, true_val)])
    accuracy = val_correct_number/features_val.shape[0] *100
    print('[%d] Evaluation acc: %.3f%%.' %(epoch + 1, accuracy))
    # save the network
    torch.save(net, RNN_FILE)

#Evaluate the accuracy on the test set
net.eval()
x = torch.tensor(features_test).float()
outputs = net(x).detach().numpy()
outputs = np.array([np.where(item == np.amax(item)) for item in outputs])
print(np.squeeze(outputs))
true_val = labels_test
print(true_val)
val_correct_number = sum([int(pred_y == target_y) for pred_y, target_y in zip(outputs, true_val)])
accuracy = val_correct_number/features_val.shape[0] *100
print('[Final] test acc: %.3f%%.' %(accuracy))
print('Finished Training')

