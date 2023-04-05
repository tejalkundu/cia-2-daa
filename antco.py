#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math





df=pd.read_csv(r'/Users/tejalkundu/Downloads/Bank_Personal_Loan_Modelling.csv')



data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = data.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1).values
Y = data['Personal Loan'].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)





class LoanModel(nn.Module):
    def __init__(self):
        super(LoanModel, self).__init__()
        self.fc1 = nn.Linear(11, 6)
        self.fc2 = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = LoanModel()




class ACO():
    def __init__(self, n_ants, n_iterations, q, evaporation_rate, alpha, beta):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.q = q
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromones = None
        self.global_best = None

    def init_pheromones(self, n_weights):
        self.pheromones = np.ones(n_weights)

    def update_pheromones(self, loss_fn):
        ant_weights = np.zeros((self.n_ants, self.pheromones.shape[0]))
        ant_fitness = np.zeros(self.n_ants)
        for i in range(self.n_ants):
            weights = np.zeros_like(self.pheromones)
            for j in range(len(weights)):
                if np.random.rand() < self.pheromones[j]:
                    weights[j] = np.random.randn()
            ant_weights[i] = weights
            weights = torch.tensor(weights).float()
            model.load_state_dict({'fc1.weight': weights[:66].reshape(6,11),
                                   'fc1.bias': weights[66:72].reshape(6),
                                   'fc2.weight': weights[72:78].reshape(1,6),
                                   'fc2.bias': weights[78]})
            ant_fitness[i] = -loss_fn(model, X_train, Y_train).item()
        for i in range(self.pheromones.shape[0]):
            delta_pheromones = 0
            for j in range(self.n_ants):
                if ant_weights[j,i] != 0:
                    delta_pheromones += (ant_fitness[j]/ant_weights[j,i])**self.q
            self.pheromones[i] = (1-self.evaporation_rate)*self.pheromones[i] + delta_pheromones

    def optimize(self, loss_fn):
        n_weights = sum(p.numel() for p in model.parameters())
        self.global_best = {'position': np.zeros(n_weights), 'fitness': -math.inf}







