import time
import pandas as pd
import tqdm

def train():
    
    cudafy = lambda x : x.cuda()
    
    DA_loss  = AssociativeLoss(visit_weight=.1)

    torch2np = lambda x : x.cpu().detach().numpy()

    train_epoch = 1000

    print('training start!')
    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()

        for (xs, ys), (xt, yt) in zip(train_svhn, train_mnist):

            xs = cudafy(xs)
            ys = cudafy(ys)
            xt = cudafy(xt)
            yt = cudafy(yt)

            losses = {}

            ### D CL training
            D_CL.zero_grad()

            phi_s, yp   = D_CL(xs)
            phi_t, ypt  = D_CL(xt)

            yp  = yp.squeeze().clone()
            ypt = ypt.squeeze().clone()

            losses['D class'] = CL_loss(yp, ys).mean()
            losses['D adapt'] = DA_loss(phi_s, phi_t, ys).mean()

            losses['D acc src']   = torch.eq(yp.max(dim=1)[1], ys).sum().float()  / batch_size_s
            losses['D acc tgt']   = torch.eq(ypt.max(dim=1)[1], yt).sum().float() / batch_size_t

            (losses['D class'] + losses['D adapt']).backward()
            DCL_optimizer.step()

            losses = { k : v.cpu().data.detach().numpy() for k, v in losses.items()}
            losses['batch'] = num_iter
            train_hist.append(losses)

            num_iter += 1

            if num_iter % 10 == 0:
                df = pd.DataFrame(train_hist)

                df.to_csv('log-discriminator/losshistory.csv')

                acc_s = df['D acc src'][-100:].mean()
                acc_t = df['D acc tgt'][-100:].mean()

                print('Epoch {}, Iteration {} - S {:.3f} % - T {:.3f} %'.format(epoch, num_iter,acc_s*100,acc_t*100), end='\r')

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time