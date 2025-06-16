"""\u6d4b\u8bd5\u4e0e\u91cd\u6784\u9650\u4ef7\u8ba2\u5355\u7c3f\u7684\u5de5\u5177\u6587\u4ef6\u3002"""
import numpy as np
import pandas as pd
import os
import sys
import tarfile
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchdiffeq import odeint as odeint
import torch.optim as optim
import logging
import utils
import models
import argparse
import scipy
from numpy import mean
from sklearn.metrics import r2_score
from models import OdeNet,DiffeqSolver, ODEFunc
import torchode as to

def main():
    """\u52a8\u73a9\u670d\u7528\u4e8e\u5c55\u793a\u5df2\u8bad\u7ec3\u6a21\u578b\u7684\u8bc4\u4f30\u7ed3\u679c\uff0c\u5e76\u5728\u6d4b\u8bd5\u96c6\u4e0a\u91cd\u6784\u8ba2\u5355\u7c3f\u3002"""
    module = 'ode'
    dataset = 'JPM'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = 64
    ode_func_net = OdeNet(n_features, 64)

    rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
    ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)
    model = models.LOBRM(module, 60, WS=True, HC=True, ES=True, device=device,diffeq_solver=ode_solver if module=='ode' else None,
                         size_latent=64, size_latent_WS=16, time=False, n_labels=4).to(device)

    five_folds_stat = False

    # \u5206\u522b\u8bc4\u4f30\u4e70\u76f8\u548c\u5356\u76f8\u7684\u7ed3\u679c
    for side in ['ask','bid']:
        # \u52a0\u8f7d\u9884\u5904\u7406\u597d\u7684\u6570\u636e
        record_list = torch.load('./parsed_data_/data_{}_{}_ws005_test_1day_1interval_timezscore_together_4exactlabel_sparse_100_new.pt'.format(side,dataset))
        data_obj = utils.parse_datasets(device,batch_size=512,dataset_train=None,dataset_val=None,dataset_test=record_list,train_mode=False)
        r_list = []
        for seed in range(1):
            # \u52a0\u8f7d\u9884\u8bad\u7ec3\u597d\u7684\u6a21\u578b\u53c2\u6570
            model.load_state_dict(torch.load('./checkpoints/{}_{}_{}_{}_True_True.ckpt'.format(dataset,side,module,seed))['state_dict'])
            label_predictions_ls = None
            label_real_ls = None

            for i in range(data_obj['n_test_batches']):

                batch = utils.get_next_batch(data_obj['test_dataloader'])  # \u53d6\u51fa\u6d4b\u8bd5\u6570\u636e
                extra_info = model.get_reconstruction(batch['data'], batch['time_steps'], batch['mask'], side=side, time=False)
                label_prediction = extra_info['label_predictions'].cpu().detach().numpy()
                labels = batch['labels'].cpu().detach().numpy()

                if i == 0:
                    label_predictions_ls = label_prediction
                    label_real_ls = labels
                else:
                    label_predictions_ls = np.vstack((label_predictions_ls,label_prediction))
                    label_real_ls = np.vstack((label_real_ls,labels))

            # \u5982\u9700\u5c06\u7ed3\u679c\u5206\u62105\u6298\u5e76\u7edf\u8ba1
            if five_folds_stat:
                folds = len(label_predictions_ls)//5
                residue = len(label_predictions_ls)%5
                if residue:
                    label_predictions_ls = label_predictions_ls[:-residue].reshape(5,folds,4)
                    label_real_ls = label_real_ls[:-residue].reshape(5,folds,4)
                else:
                    label_predictions_ls = label_predictions_ls.reshape(5,folds,4)
                    label_real_ls = label_real_ls.reshape(5,folds,4)
                label_predictions_ls = np.mean(label_predictions_ls,axis=1)
                label_real_ls = np.mean(label_real_ls,axis=1)
                delta = np.mean(abs(label_real_ls - label_predictions_ls), axis=1)
            # \u6839\u636e\u6a21\u578b\u9884\u6d4b\u548c\u771f\u5b9e\u6807\u7b7e\u8fdb\u884c\u7ebf\u6027\u62df\u5408\uff0c\u8ba1\u7b97R\u5ea6\u6570
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label_predictions_ls.reshape(-1), label_real_ls.reshape(-1))
            print('R_squared value for {} side prediction is {}'.format(side,r_value**2))
            r_list.append(r_value**2)

            # \u5c06\u6a21\u578b\u8f93\u51fa\u8fdb\u884c\u53cd\u6807\u51c6\u5316\uff0c\u4ee5\u4fbf\u548c\u539f\u59cb\u6570\u636e\u5bf9\u6bd4
            mean = np.load('./statistics/{}_{}_mean_at_each_level_2.npy'.format(dataset,side))
            std = np.load('./statistics/{}_{}_std_at_each_level_2.npy'.format(dataset,side))
            label_predictions_ls = (label_predictions_ls * std) + mean

            if side == 'ask':
                label_predictions_ask = label_predictions_ls
            else:
                label_predictions_bid = label_predictions_ls

    path_lob_train = './LOBSTER/{}_orderbook_part_5.csv'.format(dataset)
    path_msb_train = './LOBSTER/{}_message_part_5.csv'.format(dataset)
    lob, transaction = utils.time_parser(path_lob_train, path_msb_train)

    # \u6839\u636e\u9884\u6d4b\u7ed3\u679c\u91cd\u6784\u6574\u4e2aLOB\u7684\u5206\u5c42\u4fe1\u606f
    recreated_lob = pd.DataFrame(np.zeros(shape=(len(label_predictions_ask), 21)))
    recreated_length = len(label_predictions_ask)
    for i in range(recreated_length):
        # \u4fdd\u7559\u5e02\u573a\u6263\u9664\u540e\u7684timestamp
        recreated_lob.iloc[-(i + 1), 20] = lob.iloc[-(i + 1), 20]
    print('Done')
    # \u8981\u5c06\u539f\u672cLOB\u4e2d\u5df2\u77e5\u7684\u4ef7\u4f4d\u548c\u91cf\u76f4\u63a5\u62f7\u8d1d\u8fc7\u6765
    recreated_lob.iloc[:, [0, 1, 2, 3]] = lob.iloc[-recreated_length:, [0, 1, 2, 3]].values
    # \u4e3a\u4e70\u76f8\u4e0e\u5356\u76f8\u7684\u91cf\u6dfb\u52a0\u9884\u6d4b\u503c
    recreated_lob.iloc[:, [5, 9, 13, 17]] = (100 * label_predictions_ask).round()
    recreated_lob.iloc[:, 4] = recreated_lob.iloc[:, 0] + 100
    recreated_lob.iloc[:, 8] = recreated_lob.iloc[:, 0] + 200
    recreated_lob.iloc[:, 12] = recreated_lob.iloc[:, 0] + 300
    recreated_lob.iloc[:, 16] = recreated_lob.iloc[:, 0] + 400
    recreated_lob.iloc[:, [7, 11, 15, 19]] = (100 * label_predictions_bid).round()
    recreated_lob.iloc[:, 6] = recreated_lob.iloc[:, 2] - 100
    recreated_lob.iloc[:, 10] = recreated_lob.iloc[:, 2] - 200
    recreated_lob.iloc[:, 14] = recreated_lob.iloc[:, 2] - 300
    recreated_lob.iloc[:, 18] = recreated_lob.iloc[:, 2] - 400
    # \u4fdd\u5b58\u91cd\u6784\u540e\u7684LOB\u5230CSV
    recreated_lob.to_csv('./fake_lob/fake_lob_{}.csv'.format(dataset), header=True)
    # \u540c\u65f6\u5b58\u50a8\u5b9e\u9645\u7684LOB\u4fbf\u4e8e\u6bd4\u5bf9
    lob.iloc[-len(recreated_lob):,:20].to_csv('./fake_lob/real_lob_{}.csv'.format(dataset),header=True)
    print('done')
if __name__ == '__main__':
    main()
