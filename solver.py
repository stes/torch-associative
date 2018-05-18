import time
import pandas as pd
import tqdm

import torch
from torch import nn

import models

def fit(model, optim, dataset):
    
    train, val = dataset
    
    cudafy = lambda x : x.cuda()
    torch2np = lambda x : x.cpu().detach().numpy()
    
    DA_loss  = models.AssociativeLoss(visit_weight=.1)
    CL_loss  = nn.CrossEntropyLoss()
    
    cudafy(model)

    train_epoch = 1000

    print('training start!')
    start_time = time.time()
    
    num_iter = 0
    train_hist = []
    pbar_epoch = tqdm.tqdm(range(train_epoch))
    tic = time.time()
    for epoch in pbar_epoch:
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()

        pbar_batch = tqdm.tqdm(zip(train, val))
        for (xs, ys), (xt, yt) in pbar_batch:

            xs = cudafy(xs)
            ys = cudafy(ys)
            xt = cudafy(xt)
            yt = cudafy(yt)

            losses = {}

            ### D CL training
            model.zero_grad()

            phi_s, yp   = model(xs)
            phi_t, ypt  = model(xt)

            yp  = yp.squeeze().clone()
            ypt = ypt.squeeze().clone()

            losses['D class'] = CL_loss(yp, ys).mean()
            losses['D adapt'] = DA_loss(phi_s, phi_t, ys).mean()

            losses['D acc src']   = torch.eq(yp.max(dim=1)[1], ys).sum().float()  / train.batch_size
            losses['D acc tgt']   = torch.eq(ypt.max(dim=1)[1], yt).sum().float() / val.batch_size

            (losses['D class'] + losses['D adapt']).backward()
            optim.step()

            losses = { k : v.cpu().data.detach().numpy() for k, v in losses.items()}
            losses['batch'] = num_iter
            train_hist.append(losses)

            num_iter += 1

            if num_iter % 10 == 0:
                df = pd.DataFrame(train_hist)

                df.to_csv('log/losshistory.csv')

                acc_s = df['D acc src'][-100:].mean()
                acc_t = df['D acc tgt'][-100:].mean()

                pbar_batch.set_description('Epoch {}, Iteration {} - S {:.3f} % - T {:.3f} %'.format(epoch, num_iter,acc_s*100,acc_t*100))
                
            if time.time() - tic > 60:
                
                tic = time.time()
                
                torch.save(model, 'log/log-orig/model-ep{}.pth'.format(epoch))
                

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time