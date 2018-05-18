import torch

import solver, model, data

if __name__ == '__main__':
    cudafy = lambda x : x.cuda()

    # training parameters
    lr = 0.0002
    train_epoch = 20
    train_hist = []
    num_iter = 10

    # network
    D_CL  = discriminator()
    D_CL.weight_init(mean=0.0, std=0.02)

    cudafy(D_CL)

    # Binary Cross Entropy loss
    CL_loss  = nn.CrossEntropyLoss()


    # Adam optimizer
    DCL_optimizer = torch.optim.Adam(D_CL.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True)