import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder_onehot(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Encoder_onehot, self).__init__()
        
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
                
        # self.model = nn.Sequential(
        #     nn.Linear(self.data_dim, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, 2**8),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**8, 2**8),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**8, 2**6),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**6, self.hidden_dim)
        #     # nn.Tanh()
        # )
        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, 2**8),
            nn.ReLU(),
            # nn.BatchNorm1d(2**8),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.hidden_dim),
            nn.Tanh()
        )


        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Decoder_onehot(nn.Module):
    def __init__(self, hidden_dim, data_dim):
        super(Decoder_onehot, self).__init__()
    
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
        
        # self.model = nn.Sequential(
        #     nn.Linear(self.data_dim, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, 2**8),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**8, 2**8),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**8, 2**6),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(2**6, self.hidden_dim)
        #     # nn.Tanh()
        # )

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, 2**8),
            nn.ReLU(),
            # nn.BatchNorm1d(2**8),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.data_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out
    
class Encoder_client_emb(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Encoder_client_emb, self).__init__()
        
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
                
        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Decoder_client_emb(nn.Module):
    def __init__(self, hidden_dim, data_dim):
        super(Decoder_client_emb, self).__init__()
    
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.data_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out

class Encoder(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
                
        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Decoder(nn.Module):
    def __init__(self, hidden_dim, data_dim):
        super(Decoder, self).__init__()
    
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.data_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out
    

class Encoder_cont_emb(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Encoder_cont_emb, self).__init__()
        
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
                
        # self.model = nn.Sequential(
        #     nn.Linear(self.data_dim, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, 2**8),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**8, 2**8),
        #     nn.LeakyReLU(0.1),
        #     # nn.BatchNorm1d(2**8),
        #     nn.Linear(2**8, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, self.hidden_dim),
        #     nn.Tanh()
        # )

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Decoder_cont_emb(nn.Module):
    def __init__(self, hidden_dim, data_dim):
        super(Decoder_cont_emb, self).__init__()
    
        self.hidden_dim = hidden_dim #the size of latent space
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, self.data_dim),
            nn.Tanh()
        )

        # self.model = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, 2**8),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**8, 2**8),
        #     nn.LeakyReLU(0.1),
        #     # nn.BatchNorm1d(2**8),
        #     nn.Linear(2**8, 2**6),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(2**6, self.data_dim),
        #     nn.Tanh()
        # )
        
    def forward(self, x):
        out = self.model(x)
        return out