from torch import nn

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]


    if act: layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )

class SVHNmodel(nn.Module):
    
    def __init__(self):
        
        super(SVHNmodel, self).__init__()

        self.features = nn.Sequential(
            nn.InstanceNorm2d(3),
            conv2d(3,  32, 3),
            conv2d(32, 32, 3),
            conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2, padding=0), #14
            conv2d(32, 64, 3),
            conv2d(64, 64, 3),
            conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2, padding=0), # 7
            conv2d(64, 128, 3),
            conv2d(128, 128, 3),
            conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2, padding=0) # 3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 10)
        )
        
    def forward(self, x):
        
        phi  = self.features(x)
        phi_mean = phi.view(-1, 128, 16).mean(dim=-1)
        phi = phi.view(-1,128*4*4)
        y = self.classifier(phi)
        
        return phi_mean, y