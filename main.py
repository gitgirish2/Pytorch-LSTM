import argparse
import numpy as np
import os
from os.path import join, exists
import sys
import time
import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import transforms
import torch.nn as nn
from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loaders import load_data, ReplenishmentDataset
from models import ISD_RNN_MLP
from arguments import get_args

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def masked_loss(pred_g, g, warm_up=14):
    """should take prediction of size 2xseq_len-1,1 and  g of size 2xseq_len,1
     and compute the mean squared error note that we are only doing next_step predictions"""
    #extracts the predictions of warm_up+1 till end and flattens them, note that predictions are done at the previous step
    #so :,warm_up,: stores the predicted value for warm_up+1
    y_hat = pred_g[:, warm_up-1:-1].contiguous().view(-1)
    #extracts the ground truth targets from warm_up+1 till the end and flattens them
    y = g[:, warm_up:].contiguous().view(-1)
    #extracts the ground truth  from warm_up till the end-1 and flattens them
    y_1 = g[:, warm_up-1:-1].contiguous().view(-1)
    #computes loss for predictions from network
    loss = F.mse_loss(y_hat, y, reduction='mean')
    #computes loss for y_t+1=y_t
    loss_vanilla = F.mse_loss(y_1, y, reduction='mean')
    return loss, loss_vanilla

def rnn_epoch(epoch, train):
    if train:
        #sets the network in train mode so parameters accepts gradient, if there is batch_norm the batch_mean/std are used
        #if there is dropout a mask is sampled
        net.train()
        loader = train_loader
    else:
        #sets the network in test mode parameters do not have gradients, batch norm uses running moments
        # dropout outputs are rescaled since they are not sampled
        net.eval()
        loader = test_loader

    # Cycle through the data loader for num_batches equal to __len of the dataset
    cum_loss, cum_loss_van = 0, 0
    e_start = time.time()
    for i, data in enumerate(loader):
        #extract batch and gets source and targets for predictions
        #t_real is expected to be a torch float tensor batch_size,seq_lenx2,num_t_real_features
        #t_int is expected to be a torch long tensor batch_size,seq_lenx2,num_t_int_features
        #s_real is expected to be a torch float tensor batch_size,num_s_real_features
        #s_int is expected to be a torch long tensor batch_size,num_s_int_features
        start = time.time()
        t_real, t_int, s_real, s_int, g = [arr.to(device) for arr in data]

        #forward pass
        optimizer.zero_grad() #removes gradients from the optimizer, default is to accumulate
        pred_g = net(t_real, t_int, s_real, s_int) #forward pass, it outputs a 2xseq_len predictions
        #loss computation for real network and pred_g=g_(t-1)
        loss, loss_vanilla = masked_loss(pred_g, g, warm_up=args.seq_len)
        #backward
        if train:
            loss.backward() #loss is backpropagated
            optimizer.step() #parameters updates is applied
        #accumulate losses
        cum_loss += loss.item()
        cum_loss_van += loss_vanilla.item()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] LossRNN: %.6f, LossVAN: %.6f, Time: %.6f'
                % (epoch, args.epochs, i+1,
                   (len(loader.dataset) // args.batch_size) + 1,
                   cum_loss/(i + 1),
                   cum_loss_van/(i + 1),
                   time.time() - start))
        sys.stdout.flush()

    print('\nTotal epoch time:', time.time() - e_start)

    return cum_loss/(i + 1)

def debug_collate(batch):
    print("WTF")
    print(len(batch))
    for ii in range(len(batch)):
        sample = batch[ii]
#         print('sample {}'.format(ii))
#         assert len(sample) == 5
#         print('seq_real: {} -- seq_int: {} -- stat_real: {} -- stat_int: {} -- targets: {}'
#               .format(sample[0].size(), sample[1].size(), sample[2].size(), sample[3].size(), sample[4].size()))

if _name_ == "_main_":
    print("YOOO")
    #get arguments usage is python main.py --arg-1 arg1value , .. --arg-n argnvalue
    args = get_args()
    print(args)
    #check if there is a gpu in case sets it as device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #device = torch.device('cpu')
    #here something is loading the (possibly already sharded) data structures:

#     if not os.path.exists(os.path.join(args.data_dir,'ISDdata.pt')):
#         sampled_data = load_raw_data(data_folder=args.data_dir)
#     else:
#         sampled_data = None
    ISD, SD, S, I , D = load_data(args.data_dir)
    #generete the datset objects from the datasets structures and the cv_out, here using None indicates that we use the last two seq_len as test set
    dataset_train = ReplenishmentDataset(ISD, D, SD, I, S, args.seq_len, cv=None, train=True)
    #dataset_train.debug()
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=48)
    dataset_test  = ReplenishmentDataset(ISD, D, SD, I, S, args.seq_len, cv=None, train=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=16)

    print('Final train sampled %:', len(dataset_train)/len(dataset_train.ISD))
    #print(len(dataset_test.ISD) - len(dataset_train) - len(dataset_test))
#     for ii, data in enumerate(train_loader):
#         pass

    #print(dataset_train.emb_dict)

    # TODO: Rewrite gathering of seq/stat embed keys so that we don't need to force get item to set dataset's seq/stat_int_dict
    _ = dataset_train[0]

# ################# EMBEDDINGS ################

    # Dynamic embeddings
    in_seq_list = []   # each item is num of unique input values for embedding
    out_seq_list = []  # each item is num of output values for embedding

    # Static embeddings
    in_stat_list = []  # each item is num of unique input values for embedding
    out_stat_list = []  # each item is num of output values for embedding


    # FIX: seq_int_dict - only contains [0, 1] instead of [0, 1,...,10]
    for key in dataset_train.seq_int_dict.keys():
        in_seq_list.append(len(dataset_train.emb_dict[dataset_train.seq_int_dict[key]]))
    out_seq_list = [args.embs_size] * len(in_seq_list)

    for key in dataset_train.stat_int_dict.keys():
        in_stat_list.append(len(dataset_train.emb_dict[dataset_train.stat_int_dict[key]]))
    out_stat_list = [args.embs_size] * len(in_stat_list)

    in_seq_real = dataset_train.num_seq_real  # num of dynamic non-embedded features
    in_stat_real = dataset_train.num_stat_real  # num of static non-embedded features

    print('inputs to net', in_seq_list, out_seq_list, in_stat_list, out_stat_list, in_seq_real, in_stat_real, args.in_rnn, args.out_rnn, args.out_mlp)

# #################################

    # create the network the network currently lacks a method to pass in inputs but it will need to know:
    # sizes of input and outputs for each of the embeddings,
    # size of outputs and number of layers for encoders combining real and int features for rnn and mlp
    # size of output for LSTM
    # number of layers for decoder predicting the target

    net = ISD_RNN_MLP(in_seq_list, out_seq_list, in_stat_list, out_stat_list, in_seq_real, in_stat_real, args.in_rnn, args.out_rnn, args.out_mlp).to(device)

    print(next(net.parameters()).is_cuda)

    # Set up the optimizer to work on the network parameters
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=.9)
    # Scheduler and early stopper, scheduler specifies how do reduce lr once the accuracy saturates
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # Early stopper stops the job after there is no increase for patience epochs
    early_stopper = EarlyStopping('min', patience=10)
    # Actual training loop
    cur_best = None

    #rnn_epoch(0, train=True)

    for e in range(args.epochs):
        #for each epoch runs one loop on train and one validation
        rnn_epoch(e, train=True)
        with torch.no_grad():
            test_loss = rnn_epoch(e, train=False)
        #check if new test_loss is the current best
        is_best = not cur_best or test_loss < cur_best  # TODO: finish this
        if is_best:
            cur_best = test_loss
        #saves the last model and in case it is the best one it calls it best.tar
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': early_stopper.state_dict(),
            'precision': test_loss,
            'epoch': e
        }, is_best,
           join(args.save_dir, 'checkpoint.tar'),
           join(args.save_dir, 'best.tar'))
        #updates scheduler and early stopping with the new loss and eventually interrupt the job
        scheduler.step(test_loss)
        early_stopper.step(test_loss)
        if early_stopper.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break
