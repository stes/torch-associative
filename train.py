import torch

import solver, models, data

if __name__ == '__main__':

    # training parameters
    lr = 0.0002
    train_epoch = 20
    num_iter = 10

    # network
    D_CL  = models.discriminator()


    # Adam optimizer
    DCL_optimizer = torch.optim.Adam(D_CL.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True)
    
    # Dataset
    
    dataset = data.load_dataset(path="data")
    
    solver.fit(D_CL, DCL_optimizer, dataset)