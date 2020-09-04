import torch
from torch import nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(16, 4, 3, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.Flatten(),
                                    nn.Linear(3136,1000),
                                  nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                  nn.Linear(1000,100))
    def forward(self, x):
        z = self.encoder(x)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.l=nn.Sequential(nn.Linear(100,200),
                            nn.LeakyReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(200,490),
                            nn.LeakyReLU(),
                             nn.Dropout(0.5),
                            )        
        self.decoder=nn.Sequential(                                
                                   nn.ConvTranspose2d(10, 16, 2, stride=2),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(16, 1, 2, stride=2))
    def forward(self, x):
        return self.decoder(self.l(x).reshape(x.shape[0],10,7,7))
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(100, 50),
                                      nn.ReLU(),
          
                                      nn.Linear(50,50),
                                      nn.ReLU(),
                                    nn.Linear(50,1),
                                      nn.Sigmoid())

    def forward(self, z):
        disc = self.classifier(z)
        return disc
