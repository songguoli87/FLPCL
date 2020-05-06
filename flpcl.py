from cfg_pascal import Cfg
from generator import MyGenerator
from model_architecture_img import Net_img
from model_architecture_txt import Net_txt
from model_architecture_Z import Net_Z
import numpy as np
import os
import pickle
import scipy.io as sio
from sklearn.preprocessing import normalize
from retrieval_eval_ml import evaluate
import time
import torch
import torch.nn.functional as f
from torch.autograd import Variable
from torch.nn import SmoothL1Loss, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch import from_numpy,load,t,add
from utils import compute_mat_pow, compute_correlation, save_checkpoint, symmetric_average

if __name__ == '__main__':
    t0 = time.time()
    cfg = Cfg()
    loss_function = MSELoss()
    loss_CE = CrossEntropyLoss()
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)
    if cfg.resume:
        # Loading pre-trained weights
        if os.path.isfile(cfg.resume):
            print("=> Loading checkpoint '{}'".format(cfg.resume))
            checkpoint = load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            cfg.batch_size_train = checkpoint['batch_size_train']
            cfg.epsilon = checkpoint['epsilon']
            cfg.lr = checkpoint['lr']
            cfg.rho = checkpoint['rho']
            cfg.n_input_sizes = checkpoint['n_input_sizes']
            cfg.layer_sizes_F = checkpoint['layer_sizes_F']
            cfg.layer_sizes_G = checkpoint['layer_sizes_G']
            cfg.activ = checkpoint['activ']
            model_F = Net_img(cfg.layer_sizes_F, cfg.activ_img)                        
            model_G = Net_txt(cfg.layer_sizes_G, cfg.activ_txt)
            model_Z = Net_Z(cfg.layer_sizes_Z, cfg.activ_Z)
            model_F.load_state_dict(checkpoint['state_dict_F'])
            model_G.load_state_dict(checkpoint['state_dict_G'])
            model_Z.load_state_dict(checkpoint['state_dict_Z'])
            optimizer_F = Adam(list(model_F.parameters())+list(model_Z.parameters()), lr=cfg.lr)
            optimizer_G = Adam(list(model_G.parameters())+list(model_Z.parameters()), lr=cfg.lr)  
            
            optimizer_Z = Adam(list(model_Z.parameters()), lr=cfg.lr)
            optimizer_F.load_state_dict(checkpoint['optimizer_F'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_Z.load_state_dict(checkpoint['optimizer_Z'])

            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
            print('Model F: ', model_F)
            print('Model G: ', model_G)
            print('Model Z: ', model_Z)
        else:
            print("=> No checkpoint found at '{}'".format(cfg.resume))

    else:
        # Training model
        model_F = Net_img(cfg.layer_sizes_F, cfg.activ_img)        
        model_G = Net_txt(cfg.layer_sizes_G, cfg.activ_txt)
        model_Z = Net_Z(cfg.layer_sizes_Z, cfg.activ_Z)

        optimizer_F = Adam(list(model_F.parameters())+list(model_Z.parameters()), lr=cfg.lr)
        optimizer_G = Adam(list(model_G.parameters())+list(model_Z.parameters()), lr=cfg.lr)

        print('Model F: ', model_F)
        print('Model G: ', model_G)
        print('Model Z: ', model_Z)

        # Choosing batch at random and initializing co-variances
        batch = next(MyGenerator(cfg.dataset, 'train', cfg.feats, cfg.batch_size_train,
                                 train_mode=False))
        
        feat_for_F_init, F_init = model_F(Variable(from_numpy(batch[0]), requires_grad=True))     
        feat_for_G_init, G_init = model_G(Variable(from_numpy(batch[1]), requires_grad=True))
        Z_init = model_Z(Variable(from_numpy(batch[2]), requires_grad=True))
        # Initializing co-variances
        cfg.initialize_variances(F_init, G_init, Z_init)

        for epoch in range(cfg.start_epoch,cfg.training_epochs):
            print('Starting epoch number ',epoch)
            for batch in MyGenerator(cfg.dataset, 'train', cfg.feats, cfg.batch_size_train, train_mode=True):
                # Forward pass
                feat_for_F, F_train = model_F(Variable(from_numpy(batch[0])))                
                feat_for_G, G_train = model_G(Variable(from_numpy(batch[1])))
                Z_train = model_Z(Variable(from_numpy(batch[2])))
                # Updating co-variances
                cfg.update_variances(F_train, G_train, Z_train)
                # Computing conditional variables and co-variances
                cfg.update_conditional_variables(F_train, G_train, Z_train)
                # Computing right side of the loss
                FF_Z_inv_half = compute_mat_pow(cfg.FF_Z, -0.5, cfg.epsilon)
                GG_Z_inv_half = compute_mat_pow(cfg.GG_Z, -0.5, cfg.epsilon)
                FF_Z_inv_half = symmetric_average(FF_Z_inv_half)
                GG_Z_inv_half = symmetric_average(GG_Z_inv_half)
                # Fixing right side of the loss
                F_pred = (cfg.F_Z).mm(FF_Z_inv_half).detach()
                G_pred = (cfg.G_Z).mm(GG_Z_inv_half).detach()


                LL_sim = (Variable(from_numpy(batch[2]))).mm(t(Variable(from_numpy(batch[2]))))
                FF_sim = 0.5*(feat_for_F).mm(t(feat_for_F))                
                loss_FF_sim = torch.sum(-torch.mul(LL_sim,FF_sim)+torch.log(1.0 + torch.exp(FF_sim)))

                GG_sim = 0.5*(feat_for_G).mm(t(feat_for_G))                
                loss_GG_sim = torch.sum(-torch.mul(LL_sim,GG_sim)+torch.log(1.0 + torch.exp(GG_sim)))
                              
                batch_size_F = F_train.size()[0]
                loss_FF_ce = -torch.sum(f.log_softmax(feat_for_F, dim=1)) / batch_size_F
                batch_size_G = G_train.size()[0]
                loss_GG_ce = -torch.sum(f.log_softmax(feat_for_G, dim=1)) / batch_size_G

                loss_F = loss_function(cfg.F_Z, G_pred) + 0.0001*loss_FF_sim + 0.0001*loss_FF_ce
                loss_G = loss_function(cfg.G_Z, F_pred) + 0.0001*loss_GG_sim + 0.0001*loss_GG_ce                 

                # Checking for nan's
                if np.isnan(loss_F.data.numpy()) or np.isnan(loss_G.data.numpy()):
                    raise SystemExit('loss is Nan')
                # Reseting gradients, performing a backward pass, and updating the weights
                optimizer_F.zero_grad()
                loss_F.backward(retain_graph=True)
                optimizer_F.step()
                optimizer_G.zero_grad()
                loss_G.backward(retain_graph=True)
                optimizer_G.step()

    # Test #
    split = 'test'
    batch = next(MyGenerator(cfg.dataset, split, cfg.feats, cfg.batch_size_train,
                             train_mode=False))
    F_test = model_F(Variable(from_numpy(batch[0]), volatile=True))
    G_test = model_G(Variable(from_numpy(batch[1]), volatile=True))
    Z_test = model_Z(Variable(from_numpy(batch[2]), volatile=True))
    
    
    print('== Evaluating on ' + split + ' set ==')
    print('Correlation of F and G is: ', compute_correlation(F_test, G_test).data.numpy())
    # Evaluating on test set flpcl
    F_res, G_res = evaluate(F_test, G_test, cfg)

    # Saving test vectors
    pickle.dump({'F': F_test, 'G': G_test, 'Z': Z_test},
                open('./outputs/dpcca_b_' + cfg.comment + cfg.dname_ +
                     '_F_res_' + str(F_res) + '_G_res_' +
                     str(G_res) + '_split_' + split + '_vectors.p', 'wb'))
    print(time.time() - t0, 'seconds wall time')








