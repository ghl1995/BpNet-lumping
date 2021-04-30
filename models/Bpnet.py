import torch.nn.functional as F
import torch
from torch import nn

###network

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        '''self.linear1 = nn.Linear(100, 50)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(50, 25)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(25, 13)
        self.linear4 = nn.Linear(13, 7)
        self.linear5 = nn.Linear(7, 4)'''

        self.linear1 = nn.Linear(961, 200)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(200, 50)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(50, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, 4)

        '''self.linear1 = nn.Linear(800, 200)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(200, 50)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(50, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, 4)'''

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        # print(torch.max(x))
        y = F.softmax(x, dim=1)
        # y = F.gumbel_softmax(y, tau=0.5)# used for kmeans
        return y


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
