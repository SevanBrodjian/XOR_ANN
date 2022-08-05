import torch.nn as nn
from torch import Tensor

from Brain import XOR_brain

def fit(train_set,train_labels,epochs,lrate):
    
    brain = XOR_brain(lrate, nn.MSELoss(), 2, 1)

    losses = []
    for _ in range(epochs):
        losses.append(brain.step(train_set, train_labels))

    return losses,brain

def test():
    return 0

train_x = Tensor([[0, 0], [0,1], [1,0], [1,1]])
train_y = Tensor([0, 1, 1, 0])

if __name__ == '__main__':
    losses,brain = fit(train_x, train_y, 100, 0.1)

    # params = list(brain.parameters())

    # print('|------- Params -------|')
    # print(len(params))
    # print(params, '\n')

    input = Tensor([[1, 0]])
    out = brain(input)
    print(out)