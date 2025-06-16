"""LOB\u91CD\u6784\u6a21\u578b\u7684\u8bad\u7ec3\u811a\u672c\u3002

\u8be5\u811a\u672c\u53ef\u4ee5\u5b9a\u4e49\u8f93\u5165\u53c2\u6570\uff0c\u52a0\u8f7d\u6570\u636e\u5e93\uff0c\u521d\u59cb\u5316\u6a21\u578b\uff0c\u5e76\u5b8c\u6210\u8bad\u7ec3\u3001\u9a8c\u8bc1\u548c\u6d4b\u8bd5\u8fc7\u7a0b\u3002
"""
import numpy as np
import pandas as pd
import sys
import tarfile
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
import utils
import models
import argparse
import os
from models import OdeNet,DiffeqSolver, ODEFunc

def define_args():
    parser = argparse.ArgumentParser('LOB')
    parser.add_argument('--dataset',  type=str, default="INTC", help="dataset for the source model")
    parser.add_argument('--side', type=str, default="bid", help="ask side model or bid side model")
    parser.add_argument('--bs', type=int, default=512, help="batch size")
    parser.add_argument('--niter', type=int, default=50, help="number of iterations")
    parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate")
    parser.add_argument('--ls',  type=int, default=64, help="latent size in HC and ES")
    parser.add_argument('--ls_weight',  type=int, default=16, help="latent size in WS")
    parser.add_argument('--validate',  type=int, default=3, help="position of validate")

    parser.add_argument('--n_labels',  type=int, default=4, help="number of labels")
    parser.add_argument('--n_units',  type=int, default=64, help="number of units in all MLPs")
    parser.add_argument('--seed',  type=int, default=0, help="random seed")

    parser.add_argument('--main_module', type=str, default='attention', help="use what module")
    parser.add_argument('--time', action='store_true', default=False, help="whether to add time to feature vector or not")

    parser.add_argument('--WS', type=bool, default=False, help="whether to use weighting scheme or not")
    parser.add_argument('--HC', type=bool, default=False, help="whether to use history compiler or not")
    parser.add_argument('--ES', type=bool, default=True, help="whether to use market events simulator or not")
    return parser.parse_args()

def main(dataset='MSFT',side='bid',main_module='ode',HC=False,seed=0,gpu=0):
    """\u5b9e\u969b\u8fdb\u5165\u70b9\uff0c\u6307\u5b9a\u6570\u636e\u96c6\u548c\u6a21\u578b\u8bbe\u7f6e\u5bf9\u6a21\u578b\u8fdb\u884c\u8bad\u7ec3\u3002"""

    args = define_args()
    args.dataset = dataset
    args.side = side
    args.main_module = main_module
    args.seed = seed
    args.HC = HC
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = "./checkpoints/{}_{}_{}_{}_{}_{}_ex.ckpt".format(args.dataset,args.side,args.main_module,args.seed,args.ES,args.HC)
    log_path = "./logs/{}_{}_{}_{}_{}_{}_ex.log".format(args.dataset,args.side,args.main_module,args.seed,args.ES,args.HC)
    logger = open(log_path, "w")

    record_list_train = torch.load('./parsed_data_/data_%s_%s_ws005_train_3days_1interval_timezscore_together_4exactlabel_explicit_100_%d_new.pt' %(args.side,args.dataset,args.validate))
    record_list_val = torch.load('./parsed_data_/data_%s_%s_ws005_val_1day_1interval_timezscore_together_4exactlabel_explicit_100_%d_new.pt' % (args.side, args.dataset,args.validate))
    record_list_test = torch.load('./parsed_data_/data_%s_%s_ws005_test_1day_1interval_timezscore_together_4exactlabel_explicit_100_new.pt' %(args.side,args.dataset))
    data_obj = utils.parse_datasets(device,batch_size=args.bs,dataset_train = record_list_train, dataset_val = record_list_val, dataset_test = record_list_test, train_mode = True)
    input_dim = data_obj["input_dim"]
    num_batches = data_obj["n_train_batches"]

    if args.main_module == 'ode':

        n_features = 64
        ode_func_net = OdeNet(n_features, 64)

        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
        ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)
        model = models.LOBRM(args.main_module, input_dim, WS = args.WS, HC = args.HC, ES = args.ES, device=device, diffeq_solver=ode_solver,
                               size_latent= args.ls, size_latent_WS=args.ls_weight, time=args.time, n_labels=args.n_labels).to(device)

    else:
        model = models.LOBRM(args.main_module, input_dim, WS = args.WS, HC = args.HC, ES = args.ES, device=device, diffeq_solver=None,
                               size_latent= args.ls, size_latent_WS=args.ls_weight, time=args.time, n_labels=args.n_labels).to(device)

    train_loss = 0
    train_samples = 0
    val_loss_list = []
    # \u4f7f\u7528RMSprop\u4f5c\u4e3a\u4f18\u5316\u5668
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    # \u5faa\u73af\u591a\u500d\u6570\u636e\u6279\uff0c\u5b8c\u6210\u591a\u8f6e\u8bad\u7ec3
    for itr in range(1, num_batches * (args.niter+1)):
        optimizer.zero_grad()  # \u6bcf\u6b21\u5fa9\u4f53\u6d88\u9664\u7d2f\u79ef\u7684\u6839\u63d0
        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        # \u8fd0\u884c\u6a21\u578b\uff0c\u8ba1\u7b97\u4e00\u6279\u6570\u636e\u7684\u635f\u5931
        train_res = model.compute_all_losses(batch_dict, args.side, args.time)
        train_res["l1_loss"].backward()  # \u56de\u4f20\u8ba1\u7b97\u6bcf\u4e2a\u53d8\u91cf\u7684\u5e73\u5747\u6839\u63d0
        optimizer.step()  # \u66f4\u65b0\u6a21\u578b\u53c2\u6570
        train_loss = train_loss + train_res['l1_loss']*len(batch_dict['data'])
        train_samples = train_samples + len(batch_dict['data'])
        n_iters_to_val = 1
        # \u6bcf\u4e00\u8f6e\u8bad\u7ec3\u540e\u8fdb\u884c\u9a8c\u8bc1
        if itr % (n_iters_to_val * num_batches) == 0:
            # \u9a8c\u8bc1\u968f\u673a\u62bd\u53d6\u7684\u6a21\u578b\uff0c\u4e0d\u9700\u8981\u5012\u50a8\u6d1e
            with torch.no_grad():
                val_res = utils.compute_loss_all_batches(model, data_obj["val_dataloader"],
                                                          n_batches=data_obj["n_val_batches"],side = args.side,time= args.time)
                logger.write("Epoch {:04d}\n".format(itr//num_batches))
                logger.write("Train l1 loss: {:.6f}\n".format((train_loss / train_samples).detach()))
                logger.write('Validation l1 Loss {:.6f}\n'.format(val_res["l1_loss"].detach()))
                print("Train l1 loss: {:.6f}".format((train_loss / train_samples).detach()))
                print('Validation l1 Loss {:.6f}'.format(val_res["l1_loss"].detach()))
                train_loss = 0
                train_samples = 0
                val_loss_list.append(val_res['l1_loss'])

                # \u5982\u679c\u9a8c\u8bc1\u96c6\u635f\u5931\u6700\u4f4e\uff0c\u4fdd\u5b58\u5f53\u524d\u6a21\u578b
                if val_loss_list[-1] == min(val_loss_list):
                    torch.save({'state_dict': model.state_dict(),}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    with torch.no_grad():
        val_res = utils.compute_loss_all_batches(model, data_obj["val_dataloader"],
                                                  n_batches=data_obj["n_val_batches"],side = args.side,time= args.time)
        message = 'Final Val l1 Loss {:.6f}\n'.format(val_res["l1_loss"].detach())
        logger.write(message)
        print('Final Val l1 Loss {:.6f}'.format(val_res["l1_loss"].detach()))
        test_res = utils.compute_loss_all_batches(model, data_obj["test_dataloader"],
                                                  n_batches=data_obj["n_test_batches"],side = args.side,time= args.time)
        # \u6700\u7ec8\u5728\u6d4b\u8bd5\u96c6\u4e0a\u8ba1\u7b97\u5f97\u5230\u7684\u635f\u5931
        message = 'Final Test l1 Loss {:.6f}\n'.format(test_res["l1_loss"].detach())
        logger.write(message)
        print('Final Test l1 Loss {:.6f}'.format(test_res["l1_loss"].detach()))
        logger.close()


if __name__ == '__main__':
    main()
