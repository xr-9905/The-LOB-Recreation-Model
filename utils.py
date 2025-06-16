"""\u6570\u636e\u52a0\u8f7d\u3001\u8bad\u7ec3\u548c\u6d4b\u8bd5\u7684\u5de5\u5177\u51fd\u6570\u96c6"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import model_selection
import torch.nn as nn
from torchdiffeq import odeint as odeint
import torch.optim as optim
import logging
from torch.nn.modules.rnn import RNNCell, GRUCell, LSTMCell

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def variable_time_collate_fn(batch, device=torch.device("cuda"), data_type="train",
                             data_min=None, data_max=None):
    """\u5c06\u591a\u6761\u65f6\u95f4\u5e8f\u5217\u6570\u636e\u7ec4\u5408\u6210\u7b2c\u4e00\u4e2a\u4f7f\u7528\u7684\u8f93\u5165"""
    D = batch[0][2].shape[1]
    T = batch[0][2].shape[0]
    combined_tt = torch.zeros([len(batch),100]).to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), T, D]).to(device)
    combined_mask = torch.zeros([len(batch), T, D]).to(device)

    N_labels = 4

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        '''for explicit data'''
        vals[:,[0,2]] = (vals[:,[0,2]] - vals[-1,[0,2]]) / 10
        vals[:,4] = 0

        combined_tt[b] = torch.cumsum(torch.cat((torch.tensor([0]).to(device),torch.ceil(torch.log(tt[1:]-tt[:-1]+1.001)))),dim=0).float()
        combined_vals[b] = vals
        combined_mask[b] = mask

        if labels is not None:
            combined_labels[b] = labels

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    return data_dict

def parse_datasets(device,batch_size,dataset_train,dataset_val,dataset_test,train_mode = True):
    """\u751f\u6210\u7528\u4e8e\u8bad\u7ec3\u548c\u6d4b\u8bd5\u7684DataLoader"""

    # Shuffle and split
    if train_mode:

        train_data = dataset_train
        val_data = dataset_val
        test_data = dataset_test
    else:
        test_data = dataset_test

    record_id, tt, vals, mask, labels = test_data[0]
    input_dim = vals.size(-1)
    if train_mode:
        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device, data_type = "train",
            data_min = None, data_max = None))
        val_dataloader =   DataLoader(val_data, batch_size= batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device, data_type = "validate",
            data_min = None, data_max = None))
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device, data_type = "test",
            data_min = None, data_max = None))
    else:
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device, data_type = "test",
            data_min = None, data_max = None))
    if train_mode:
        data_objects = {
                        "train_dataloader": inf_generator(train_dataloader),
                        "val_dataloader": inf_generator(val_dataloader),
                        "test_dataloader": inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_val_batches": len(val_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "n_labels": 4}
    else:
        data_objects = {
                        "test_dataloader": inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_test_batches": len(test_dataloader),
                        "n_labels": 4}
    return data_objects

def inf_generator(iterable):
    """\u4f7f\u5f97DataLoader\u80fd\u5728\u65e0\u9650\u7684\u5faa\u73af\u4e2d\u4f7f\u7528"""
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def compute_l1_loss(label_predictions, true_label):
    """\u8ba1\u7b97L1\u7a0b\u5ea6\u7684\u635f\u5931"""

    if (len(label_predictions.size()) == 3):
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

    assert(not torch.isnan(label_predictions).any())
    assert(not torch.isnan(true_label).any())

    true_label = true_label.repeat(n_traj_samples, 1, 1)

    label_predictions = (label_predictions.reshape(n_traj_samples * n_traj * n_tp, n_dims))
    true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

    l1_loss = nn.L1Loss()(label_predictions, true_label)

    return l1_loss

def create_net(n_inputs, n_outputs, n_layers = 1,
    n_units = 64, nonlinear = nn.Tanh):
    """\u6784\u9020\u7b80\u5355\u7684MLP\u7f51\u7edc"""
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

class ODEFunc(nn.Module):
    def __init__(self, ode_func_net):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()
    return data_dict


def compute_loss_all_batches(model,test_dataloader,n_batches,side,time=False):
    total = {}
    test_loss = 0
    num_samples = 0
    for itr in range(n_batches):
        test_batch_dict = get_next_batch(test_dataloader)
        test_res = model.compute_all_losses(test_batch_dict,side,time)['l1_loss']
        test_loss = test_loss + test_res * len(test_batch_dict['data'])
        num_samples = num_samples + len(test_batch_dict['data'])
    total['l1_loss'] = test_loss / num_samples

    return total

def time_parser(path_lob, path_msb):
    with open(path_lob) as f:
        data_lob = pd.read_csv(f, header=None)
    with open(path_msb) as f:
        data_msb = pd.read_csv(f, header=None, names=['time', 'type', 'index', 'quantity', 'price', 'direction'])
    time_index = data_msb['time']
    timeline = time_index.unique()
    timesteps = len(timeline)
    lob_with_time = data_lob.join(time_index)
    data_transaction = data_msb.loc[(data_msb['type'] == 4), ['time', 'price', 'quantity', 'direction']]
    transaction = []
    for time in data_transaction['time'].unique():
        target = data_transaction.loc[data_transaction['time'] == time]
        transaction_sum = sum(target['price'] * target['quantity'])
        transaction_volumn = sum(target['quantity'])
        if sum(target['direction'])>=0:
            direction = 1
        else:
            direction = -1
        weighted_price = round(transaction_sum / transaction_volumn / 100) * 100
        transaction.append([time,weighted_price,transaction_volumn,direction])
    transaction = pd.DataFrame(np.array(transaction),columns = ['time', 'price', 'quantity', 'direction'])
    transaction.index = range(len(transaction))
    lob = pd.DataFrame(np.zeros(shape=(len(transaction), 20)))
    for i in range(len(transaction)):
        lob.iloc[i, :] = lob_with_time.loc[(lob_with_time['time'] == transaction.iloc[i,0])].iloc[-1,:-1]
    lob = lob.join(transaction[['time']])
    return lob,transaction

def fetch_inputs(A,B,m,h,k,device):
    # Repeat B along the third dimension
    B_repeated = B.unsqueeze(-1).repeat(1, 1, k)

    # Flatten A and B_repeated into 1D tensors
    A_flat = A.reshape(-1)
    B_flat = B_repeated.reshape(-1)

    # Use masked_select to select elements from A_flat based on B_flat
    selected = torch.masked_select(A_flat, B_flat)

    # Calculate the sizes of the segments based on B
    sizes = (B.sum(dim=1) * k).tolist()

    # Split the selected elements into segments
    segments = selected.split(sizes)

    # Pad each segment to have a length of h*k and stack them to form C
    C = torch.stack([torch.cat([seg, torch.zeros(h * k - seg.numel()).to(device)]) for seg in segments]).view(m, h, k)

    return C


