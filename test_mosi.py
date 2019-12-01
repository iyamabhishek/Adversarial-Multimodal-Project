import numpy as np
seed = 123
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import cPickle as pickle
import time
import json, os, ast, h5py

from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys

class MFN(nn.Module):
    def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
        super(MFN, self).__init__()
        [self.d_m1,self.d_m2,self.d_m3] = config["input_dims"]
        [self.dh_m1,self.dh_m2,self.dh_m3] = config["h_dims"]
        total_h_dim = self.dh_m1+self.dh_m2+self.dh_m3
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        self.lstm_m1 = nn.LSTMCell(self.d_m1, self.dh_m1)
        self.lstm_m2 = nn.LSTMCell(self.d_m2, self.dh_m2)
        self.lstm_m3 = nn.LSTMCell(self.d_m3, self.dh_m3)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        
    def forward(self,x):
        x_m1 = x[:,:,:self.d_m1]
        x_m2 = x[:,:,self.d_m1:self.d_m1+self.d_m2]
        x_m3 = x[:,:,self.d_m1+self.d_m2:]
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]
        self.h_m1 = torch.zeros(n, self.dh_m1).cuda()
        self.h_m2 = torch.zeros(n, self.dh_m2).cuda()
        self.h_m3 = torch.zeros(n, self.dh_m3).cuda()
        self.c_m1 = torch.zeros(n, self.dh_m1).cuda()
        self.c_m2 = torch.zeros(n, self.dh_m2).cuda()
        self.c_m3 = torch.zeros(n, self.dh_m3).cuda()
        self.mem = torch.zeros(n, self.mem_dim).cuda()
        all_h_m1s = []
        all_h_m2s = []
        all_h_m3s = []
        all_c_m1s = []
        all_c_m2s = []
        all_c_m3s = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_m1 = self.c_m1
            prev_c_m2 = self.c_m2
            prev_c_m3 = self.c_m3
            # curr time step
            new_h_m1, new_c_m1 = self.lstm_m1(x_m1[i], (self.h_m1, self.c_m1))
            new_h_m2, new_c_m2 = self.lstm_m2(x_m2[i], (self.h_m2, self.c_m2))
            new_h_m3, new_c_m3 = self.lstm_m3(x_m3[i], (self.h_m3, self.c_m3))
            # concatenate
            prev_cs = torch.cat([prev_c_m1,prev_c_m2,prev_c_m3], dim=1)
            new_cs = torch.cat([new_c_m1,new_c_m2,new_c_m3], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            self.h_m1, self.c_m1 = new_h_m1, new_c_m1
            self.h_m2, self.c_m2 = new_h_m2, new_c_m2
            self.h_m3, self.c_m3 = new_h_m3, new_c_m3
            all_h_m1s.append(self.h_m1)
            all_h_m2s.append(self.h_m2)
            all_h_m3s.append(self.h_m3)
            all_c_m1s.append(self.c_m1)
            all_c_m2s.append(self.c_m2)
            all_c_m3s.append(self.c_m3)

        # last hidden layer last_hs is n x h
        last_h_m1 = all_h_m1s[-1]
        last_h_m2 = all_h_m2s[-1]
        last_h_m3 = all_h_m3s[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_m1,last_h_m2,last_h_m3,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output

def train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    X_train = X_train.swapaxes(0,1)
    X_valid = X_valid.swapaxes(0,1)
    X_test = X_test.swapaxes(0,1)

    d = X_train.shape[2]
    h = 128
    t = X_train.shape[0]
    output_dim = 1
    dropout = 0.5

    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs

    #model = EFLSTM(d,h,output_dim,dropout)
    model = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])
    #optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

    # optimizer = optim.SGD([
    #                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
    #                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
    #             ], momentum=0.9)

    criterion = nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

    def train(model, batchsize, X_train, y_train, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[1]
        num_batches = total_n / batchsize
        for batch in xrange(num_batches):
            start = batch*batchsize
            end = (batch+1)*batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[:,start:end]).cuda()
            batch_y = torch.Tensor(y_train[start:end]).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid).cuda()
            batch_y = torch.Tensor(y_valid).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            epoch_loss = criterion(predictions, batch_y).item()
        return epoch_loss

    def predict(model, X_test):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions

    best_valid = 999999.0
    rand = random.randint(0,100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print epoch, train_loss, valid_loss, 'saving model'
            torch.save(model, 'temp_models/mfn_%d.pt' %rand)
        else:
            print epoch, train_loss, valid_loss

    print 'model number is:', rand
    model = torch.load('temp_models/mfn_%d.pt' %rand)

    predictions = predict(model, X_test)
    mae = np.mean(np.absolute(predictions-y_test))
    print "mae: ", mae
    corr = np.corrcoef(predictions,y_test)[0][1]
    print "corr: ", corr
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print "mult_acc: ", mult
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print "mult f_score: ", f_score
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print "Confusion Matrix :"
    print confusion_matrix(true_label, predicted_label)
    print "Classification Report :"
    print classification_report(true_label, predicted_label, digits=5)
    print "Accuracy ", accuracy_score(true_label, predicted_label)
    sys.stdout.flush()

def test(X_test, y_test, metric):
    X_test = X_test.swapaxes(0,1)
    def predict(model, X_test):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions
    if metric == 'mae':
        model = torch.load('best/mfn_mae.pt',map_location='cuda:0')
    if metric == 'acc':
        model = torch.load('best/mfn_acc.pt',map_location='cuda:0')
    model = model.cpu().cuda()
    
    predictions = predict(model, X_test)
    print predictions.shape
    print y_test.shape
    mae = np.mean(np.absolute(predictions-y_test))
    print "mae: ", mae
    corr = np.corrcoef(predictions,y_test)[0][1]
    print "corr: ", corr
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print "mult_acc: ", mult
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print "mult f_score: ", f_score
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print "Confusion Matrix :"
    print confusion_matrix(true_label, predicted_label)
    print "Classification Report :"
    print classification_report(true_label, predicted_label, digits=5)
    print "Accuracy ", accuracy_score(true_label, predicted_label)
    sys.stdout.flush()

def load_saved_data(x_train, x_valid, x_test, y_train, y_valid, y_test):
    h5f = h5py.File(x_train,'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(y_train,'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(x_valid,'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(y_valid,'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(x_test,'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(y_test,'r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data('data/x_trimodal_train.h5', 'data/x_trimodal_valid.h5', 'data/x_trimodal_test.h5', 'data/y_train.h5', 'data/y_valid.h5', 'data/y_test.h5')

config = dict()
config["input_dims"] = [300,5,20]
hm1 = random.choice([32,64,88,128,156,256])
hm2 = random.choice([8,16,32,48,64,80])
hm3 = random.choice([8,16,32,48,64,80])
config["h_dims"] = [hm1,hm2,hm3]
config["memsize"] = random.choice([64,128,256,300,400])
config["windowsize"] = 2
config["batchsize"] = random.choice([32,64,128,256])
config["num_epochs"] = 50
config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
NN1Config = dict()
NN1Config["shapes"] = random.choice([32,64,128,256])
NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
NN2Config = dict()
NN2Config["shapes"] = random.choice([32,64,128,256])
NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
gamma1Config = dict()
gamma1Config["shapes"] = random.choice([32,64,128,256])
gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
gamma2Config = dict()
gamma2Config["shapes"] = random.choice([32,64,128,256])
gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
outConfig = dict()
outConfig["shapes"] = random.choice([32,64,128,256])
outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
print configs
train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)