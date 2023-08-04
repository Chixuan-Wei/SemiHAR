import numpy as np
import torch
import torch.nn as nn
from lab import TSR as tsr
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class UserClassifier(nn.Module):
    def __init__(self,output_size, input_size):
        super(UserClassifier, self).__init__()
        self.clssifier = nn.Sequential(
            nn.Linear(input_size,100),
            nn.ReLU(),
            nn.Linear(100,output_size),
            nn.Softmax(dim=1)
        )
        nn.init.xavier_normal_(self.clssifier[0].weight)
        nn.init.xavier_normal_(self.clssifier[2].weight)

    def forward(self,x):
        classification = self.clssifier(x)
        return classification

class LabelClassifier(nn.Module):
    def __init__(self,output_size, input_size):
        super(LabelClassifier, self).__init__()
        self.clssifier = nn.Sequential(
            nn.Linear(input_size,100),
            nn.ReLU(),
            nn.Linear(100,output_size),
            nn.Softmax(dim=1)
        )
        nn.init.xavier_normal_(self.clssifier[0].weight)
        nn.init.xavier_normal_(self.clssifier[2].weight)

    def forward(self,x):
        classification = self.clssifier(x)
        return classification

class Decoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(Decoder,self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,output_size)
        )
        nn.init.xavier_normal_(self.decode[0].weight)
        nn.init.xavier_normal_(self.decode[2].weight)

    def forward(self, bottleneck):
        decoderimage = self.decode(bottleneck)
        return decoderimage


class Encoder(nn.Module):
    def __init__(self, x_mk, x_tr, u_out, n_out, image_size):
        super(Encoder,self).__init__()

        self.distri = nn.Sequential(
             nn.Conv2d(1, 128, kernel_size=(7,3), stride=(1,3)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
             nn.Conv2d(128, 256, kernel_size=(5,1), stride=(1,1)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
             nn.Conv2d(256, 384, kernel_size=(1,1), stride=(1,1)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
             nn.Conv2d(384, 256, kernel_size=(1,1), stride=(1,1)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
        )

        h_u = output_conv_size(x_tr.shape[2],kernel_size=7,stride=1,padding=0)
        w_u = output_conv_size(x_tr.shape[3],kernel_size=3,stride=3,padding=0)
        h_u = output_conv_size(h_u,kernel_size=2,stride=2,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=5,stride=1,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=2,stride=2,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=1,stride=1,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=2,stride=2,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=1,stride=1,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        h_u = output_conv_size(h_u,kernel_size=2,stride=2,padding=0)
        w_u = output_conv_size(w_u,kernel_size=1,stride=1,padding=0)
        x_input = 256*h_u*w_u

        self.imagemk = nn.Sequential(
             nn.Conv2d(1, 128, kernel_size=(7,3), stride=(1,3)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
             nn.Conv2d(128, 256, kernel_size=(5,1), stride=(1,1)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
             nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1)),
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
             nn.ReLU(),
        )

        h_mk = output_conv_size(x_mk.shape[2],kernel_size=7,stride=1,padding=0)
        w_mk = output_conv_size(x_mk.shape[3],kernel_size=3,stride=3,padding=0)
        h_mk = output_conv_size(h_mk,kernel_size=2,stride=2,padding=0)
        w_mk = output_conv_size(w_mk,kernel_size=1,stride=1,padding=0)
        h_mk = output_conv_size(h_mk,kernel_size=5,stride=1,padding=0)
        w_mk = output_conv_size(w_mk,kernel_size=1,stride=1,padding=0)
        h_mk = output_conv_size(h_mk,kernel_size=2,stride=2,padding=0)
        w_mk = output_conv_size(w_mk,kernel_size=1,stride=1,padding=0)
        h_mk = output_conv_size(h_mk,kernel_size=1,stride=1,padding=0)
        w_mk = output_conv_size(w_mk,kernel_size=1,stride=1,padding=0)
        h_mk = output_conv_size(h_mk,kernel_size=2,stride=2,padding=0)
        w_mk = output_conv_size(w_mk,kernel_size=1,stride=1,padding=0)

        mk_input = 256*h_mk*w_mk

        self.flatten = nn.Flatten()

        self.mk_Linear = nn.Linear(in_features=mk_input, out_features=1024)
        self.x_Linear = nn.Linear(in_features=x_input, out_features=1024)

        self.attention = tsr.TaskRelation(d_model=1024, d_k=1024, d_v=1024, h=4, H=1, W=3, ratio=2, apply_transform=True)

        self.user = UserClassifier(output_size=u_out, input_size=1024)
        self.label = LabelClassifier(output_size=n_out, input_size=1024)
        self.image = Decoder(output_size=image_size, input_size=1024)

        nn.init.xavier_normal_(self.distri[0].weight)
        nn.init.xavier_normal_(self.distri[3].weight)
        nn.init.xavier_normal_(self.distri[6].weight)
        nn.init.xavier_normal_(self.imagemk[0].weight)
        nn.init.xavier_normal_(self.imagemk[3].weight)
        nn.init.xavier_normal_(self.imagemk[6].weight)
        nn.init.xavier_normal_(self.x_Linear.weight)
        nn.init.xavier_normal_(self.mk_Linear.weight)

    def forward(self, x_mk, x_labeled, x):

        mk = self.imagemk(x_mk)
        mk = self.flatten(mk)
        mk = self.mk_Linear(mk)

        u = self.distri(x)
        u = self.flatten(u)
        u = self.x_Linear(u)

        c = self.distri(x_labeled)
        c = self.flatten(c)
        c = self.x_Linear(c)

        i_feature = torch.reshape(mk,(mk.shape[0], 1, mk.shape[1]))
        u_feature = torch.reshape(u,(u.shape[0], 1, u.shape[1]))
        c_feature = torch.reshape(c,(c.shape[0], 1, c.shape[1]))
        x = torch.cat((i_feature,u_feature,c_feature),dim=1).to(device)

        att_x = self.attention(x,x,x)

        i_att_feature = att_x[:,0,:]
        u_att_feature = att_x[:,1,:]
        c_att_feature = att_x[:,2,:]

        mk = self.image(i_att_feature)
        mk_out = torch.reshape(mk,(mk.shape[0], -1, x_mk.shape[2], x_mk.shape[3]))

        u_out = self.user(u_att_feature)

        c_out = self.label(c_att_feature)

        return  mk_out,u_out,c_out

    def fortsne(self, x):

        c = self.distri(x)
        c = self.flatten(c)
        c = self.x_Linear(c)

        return c



def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output





