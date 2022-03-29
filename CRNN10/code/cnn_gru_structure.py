import torch
import torch.nn.functional as F
import numpy as np

# Set the random seed manually for reproducibility.
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Vgg_16(torch.nn.Module):

    def __init__(self):
        super(Vgg_16, self).__init__()
        self.convolution1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pooling2 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.convolution4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = torch.nn.MaxPool2d(2, stride= 2) 
        self.convolution5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(512)
        self.convolution6 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.BatchNorm2 = torch.nn.BatchNorm2d(512)
        self.pooling4 = torch.nn.MaxPool2d(2, stride= 2)
        self.convolution7 = torch.nn.Conv2d(512, 512, 3, padding= 1)
        self.BatchNorm3 = torch.nn.BatchNorm2d(512)
        self.convolution8 = torch.nn.Conv2d(512, 512, 3, padding= 1)
        self.BatchNorm4 = torch.nn.BatchNorm2d(512)
        self.pooling5 = torch.nn.MaxPool2d(2, stride= 2)


    def forward(self, x):
        x = F.relu(self.convolution1(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3(x), inplace=True)
        x = F.relu(self.convolution4(x), inplace=True)
        x = self.pooling3(x)
        x = self.convolution5(x)
        x = F.relu(self.BatchNorm1(x), inplace=True)
        # x = F.relu(x)
        x = self.convolution6(x)
        x = F.relu(self.BatchNorm2(x), inplace=True)
        # x = F.relu(x)
        x = self.pooling4(x)
        x = self.convolution7(x)
        x = F.relu(self.BatchNorm3(x), inplace=True)
        # x = F.relu(x)
        x = self.convolution8(x)
        x = F.relu(self.BatchNorm4(x), inplace=True)
        # x = F.relu(x)
        x = self.pooling5(x)
        return x  # batch * 512 * 1 * seq_len


class RNN(torch.nn.Module):
    def __init__(self, class_num, hidden_unit):
        super(RNN, self).__init__()
        self.Bidirectional_GRU1 = torch.nn.GRU(512, hidden_unit, num_layers=2, batch_first= True, dropout = 0.5, bidirectional= False)
        # self.embedding1 = torch.nn.Linear(hidden_unit * 2, 512)
        # self.Bidirectional_GRU2 = torch.nn.GRU(512, hidden_unit, bidirectional=True, batch_first= True)
        self.embedding1 = torch.nn.Linear(hidden_unit, class_num)
        # self.embedding1 = torch.nn.Linear(hidden_unit * 2, class_num)

    def forward(self, x):
        x = self.Bidirectional_GRU1(x)   # GRU output: output, hidden state(h_n)
        # b, T, h = x[0].size()   # x[0]: (batch, seq_len, num_directions * hidden_size)
        # x = self.embedding1(x[0].view(b * T, h))  # pytorch view() reshape as [b * T, nOut]
        # x = x.view(b, T, -1)  # [b, seq_len, 512]
        # x = self.Bidirectional_GRU2(x)
        b, T, h = x[0].size()
        # print(x[0].size())
        x = self.embedding1(x[0].contiguous().view(b * T, h))
        x = x.contiguous().view(b, T, -1)
        return x  # [b,seq_len,class_num]


# output: [b,s,class_num]
class CRNN(torch.nn.Module):
    def __init__(self, class_num=1, hidden_unit=256):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential()
        self.cnn.add_module('vgg_16', Vgg_16())
        self.rnn = torch.nn.Sequential()
        self.rnn.add_module('rnn', RNN(class_num, hidden_unit))

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        # print(x.size()): b,c,h,w
        assert h == 1   # "the height of conv must be 1"
        x = x.squeeze(2)  # remove h dimension, b *512 * width
        x = x.permute(0, 2, 1)  # [b, w, c] = [batch, seq_len, input_size]
        # print('rnn input size', x.size())
        # x = x.transpose(1, 2)
        x = self.rnn(x)
        return x