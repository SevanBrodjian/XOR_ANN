import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, cat
from torch.autograd import Variable

class XOR_brain(nn.Module):

    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(XOR_brain, self).__init__()

        self.loss_fn = loss_fn

        self.fc1_in = nn.Linear(in_size, 3)
        self.fc2_out = nn.Linear(3, out_size)

        self.optimizer = optim.SGD(self.parameters(), lr = lrate)

    def forward(self, x):
        y_hat = Tensor()
        for input in x:
            input = F.sigmoid(self.fc1_in(input))
            input = F.sigmoid(self.fc2_out(input))
            if(input >= 0.5):
                y_hat = cat((y_hat, Tensor([1])))
            else:
                y_hat = cat((y_hat, Tensor([0])))
        return y_hat

    def step(self, x, y):
        assert len(x) == len(y)
        y_hat = self(x)
        print(y_hat, y)
        loss = Variable(self.loss_fn(y_hat, y), requires_grad=True)

        # Optimize based on gradient from loss function
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().numpy()