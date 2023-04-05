#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math




df=pd.read_csv(r'/Users/tejalkundu/Downloads/Bank_Personal_Loan_Modelling.csv')




df.head()




df.info()




X = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1).values
Y = df['Personal Loan'].values.reshape(-1,1)
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




class PSO():
    def __init__(self, pop_size, lr, n_iterations):
        self.pop_size = pop_size
        self.lr = lr
        self.n_iterations = n_iterations
        self.inertia_weight = 0.9
        self.c1 = 2
        self.c2 = 2
        self.global_best = None
        self.particles = []

    def init_particles(self, n_weights):
        for i in range(self.pop_size):
            particle = {'position': np.random.randn(n_weights),
                        'velocity': np.zeros(n_weights),
                        'best_position': None,
                        'best_fitness': -math.inf}
            self.particles.append(particle)

    def update_particles(self, loss_fn):
        for particle in self.particles:
            weights = torch.tensor(particle['position']).float()
            model.load_state_dict({'fc1.weight': weights[:66].reshape(6,11),
                                   'fc1.bias': weights[66:72].reshape(6),
                                   'fc2.weight': weights[72:78].reshape(1,6),
                                   'fc2.bias': weights[78]})
            fitness = -loss_fn(model, X_train, Y_train).item()
            if fitness > particle['best_fitness']:
                particle['best_fitness'] = fitness
                particle['best_position'] = particle['position'].copy()
                if fitness > self.global_best['fitness']:
                    self.global_best['fitness'] = fitness
                    self.global_best['position'] = particle['position'].copy()
            particle['velocity'] = (self.inertia_weight * particle['velocity'] +
                                    self.c1 * np.random.rand(len(particle['position'])) *
                                    (particle['best_position'] - particle['position']) +
                                    self.c2 * np.random.rand(len(particle['position'])) *
                                    (self.global_best['position'] - particle['position']))
            particle['position'] += self.lr * particle['velocity']

    def optimize(self, loss_fn):
        n_weights = sum(p.numel() for p in model.parameters())
        self.global_best = {'position': np.zeros(n_weights), 'fitness': -math.inf}
        self.init






