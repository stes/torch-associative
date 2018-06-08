import h5py
import torch
from torch import nn

import torch.nn.functional as F

class WalkerLoss(nn.Module):

    def forward(self, Psts, y):
        equality_matrix = torch.eq(y.clone().view(-1,1), y).float()
        p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        p_target.requires_grad = False

        L_walker = F.kl_div(torch.log(1e-8 + Psts), p_target, size_average=False)
        L_walker /= p_target.size()[0]

        return L_walker

class VisitLoss(nn.Module):

    def forward(self, Pt):
        p_visit = torch.ones([1, Pt.size()[1]]) / float(Pt.size()[1])
        p_visit.requires_grad = False
        if Pt.is_cuda: p_visit = p_visit.cuda()
        L_visit = F.kl_div(torch.log(1e-8 + Pt), p_visit, size_average=False)
        L_visit /= p_visit.size()[0]

        return L_visit

class AssociationMatrix(nn.Module):

    def __init__(self):
        super(AssociationMatrix, self).__init__()

    def forward(self, xs, xt):
        """
        xs: (Ns, K, ...)
        xt: (Nt, K, ...)
        """

        # TODO not sure why clone is needed here
        Bs = xs.size()[0]
        Bt = xt.size()[0]

        xs = xs.clone().view(Bs, -1)
        xt = xt.clone().view(Bt, -1)

        W = torch.mm(xs, xt.transpose(1,0))

        # p(xt | xs) as softmax, normalize over xt axis
        Pst = F.softmax(W, dim=1) # Ns x Nt
        # p(xs | xt) as softmax, normalize over xs axis
        Pts = F.softmax(W.transpose(1,0), dim=1) # Nt x Ns

        # p(xs | xs)
        Psts = Pst.mm(Pts) # Ns x Ns

        # p(xt)
        Pt = torch.mean(Pst, dim=0, keepdim=True) # Nt

        return Psts, Pt

class AssociativeLoss(nn.Module):

    def __init__(self, walker_weight = 1., visit_weight = 1.):
        super(AssociativeLoss, self).__init__()

        self.matrix = AssociationMatrix()
        self.walker = WalkerLoss()
        self.visit  = VisitLoss()

        self.walker_weight = walker_weight
        self.visit_weight  = visit_weight

    def forward(self, xs, xt, y):

        Psts, Pt = self.matrix(xs, xt)
        L_walker = self.walker(Psts, y)
        L_visit  = self.visit(Pt)

        return self.visit_weight*L_visit + self.walker_weight*L_walker

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]

    if act: layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )

class SVHNmodel(nn.Module):

    """
    Model for application on SVHN data (32x32x3)
    Architecture identical to https://github.com/haeusser/learning_by_association
    """

    def __init__(self):

        super(SVHNmodel, self).__init__()

        self.features = nn.Sequential(
            nn.InstanceNorm2d(3),
            conv2d(3,  32, 3),
            conv2d(32, 32, 3),
            conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(32, 64, 3),
            conv2d(64, 64, 3),
            conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(64, 128, 3),
            conv2d(128, 128, 3),
            conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2, padding=0)
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


class FrenchModel(nn.Module):

    """
    Model used in "Self-Ensembling for Visual Domain Adaptation"
    by French et al.
    """

    def __init__(self):

        super(FrenchModel, self).__init__()

        def conv2d_3x3(inp,outp,pad=1):
            return nn.Sequential(
                nn.Conv2d(inp,outp,kernel_size=3,padding=pad),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def conv2d_1x1(inp,outp):
            return nn.Sequential(
                nn.Conv2d(inp,outp,kernel_size=1,padding=0),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def block(inp,outp):
            return nn.Sequential(
                conv2d_3x3(inp,outp),
                conv2d_3x3(outp,outp),
                conv2d_3x3(outp,outp)
            )

        self.features = nn.Sequential(
            block(3,128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128,256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d_3x3(256, 512, pad=0),
            conv2d_1x1(512, 256),
            conv2d_1x1(256, 128),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):

        phi  = self.features(x)
        phi = phi.view(-1,128)
        # print(x.size(), phi.size())
        y = self.classifier(phi)

        return phi, y
