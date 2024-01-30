import numpy as np
from matplotlib import pyplot as plt
from client_server_connection import client
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# moves your model to train on your gpu if available else it uses your cpu
device = ("cuda" if torch.cuda.is_available() else "cpu")
import math
import collections
import time
from domainbed import TSdatasets as datasets
from domainbed import TShparams_registry as hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.misc import domain_contrastive_loss as contrastive_loss_function

dataset="PRPD_GIS"
local_IED="IED_0"
batch_size=32
# steps_per_epoch = 12   # Each epoch run 12 step for each round
n_steps = 100+1   # the number of rounds
algorithm = 'FL_ERM'   # Federate learning algorithm

def get_data(dataset, local_IED):
    # Dummy data for the client
    data_dir= '/PRPD_Datasets'
    test_envs= []
    hparams= None
    holdout_fraction= 0.2
    trial_seed= 0
    # hparams['data_augmentation'] = hparams(True)
    if dataset in vars(datasets):
        dataset = vars(datasets)[dataset](data_dir,dataset,local_IED,
            test_envs, hparams)
    else:
        raise NotImplementedError
        
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*holdout_fraction),
            misc.seed_hash(trial_seed, env_i))
#     print(out.shape, in_.shape)
#     in_weights = misc.make_weights_for_balanced_classes(in_)
#     out_weights = misc.make_weights_for_balanced_classes(out)
        in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

        
    return in_splits,out_splits


while True:
    try:
        # Code for initializing and running the client
        cli = client()
        
        print("Loading data....")
        
        in_splits,out_splits = get_data(dataset, local_IED)
        steps_per_epoch = math.ceil(min([len(env)/batch_size for env,_ in in_splits]))
        algorithm_class = algorithms.get_algorithm_class(algorithm)
        algorithm = algorithm_class(dataset, (3600,128,), 5,
        1, None, 1, steps_per_epoch)

# algorithm.to(device)


# train_minibatches_iterator = [iter(loader) for loader in train_loaders]
        checkpoint_vals = collections.defaultdict(lambda: [])
        hparams= None

        last_results_keys = None
        contrastive_loss = 0
        start_step = 0
        brake=False
        local_model = algorithm
        local_model.to(device)
        local_model.train()
# for tensorboard logs
    # note test_target_in is the final test accuracy reported 
        test_envs= []
        train_source = ['env{}_in_acc'.format(i) for i in range(len(in_splits)) 
                                if (i not in test_envs)]
        test_source = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) 
                                                        if (i not in test_envs)]
        test_target_in = ['env{}_in_acc'.format(i) for i in range(len(in_splits)) 
                                                                if (i in test_envs)] 
        test_target_out = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) 
                                                        if (i in test_envs)] 
        print(train_source,test_source,test_target_in,test_target_out)

        train_loaders = [InfiniteDataLoader(
                                        dataset=env,
                                        weights=env_weights,
                                        batch_size=batch_size)
                                for i, (env, env_weights) in enumerate(in_splits)
                                    if i not in test_envs]

        train_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
#         print(train_loader_names)

        eval_loaders = [FastDataLoader(
                        dataset=env,
                            batch_size=64)
                                    for env, _ in (in_splits + out_splits)]
        eval_weights = [None for _, weights in (in_splits + out_splits)]
        eval_loader_names = ['env{}_in'.format(i)
                                    for i in range(len(in_splits))]
        eval_loader_names += ['env{}_out'.format(i)
                                                for i in range(len(out_splits))]
#         print(eval_loader_names)


  
        for step in range(start_step, n_steps):
    # prep model for training
            local_model.train()
    # model2.train()
            train_loss = 0
    # tr2=iter(trainloader2)
            train_minibatches_iterator = [iter(loader) for loader in train_loaders]
            step_start_time = time.time()
            for local_loader in train_minibatches_iterator: 

                step_vals = local_model.update(local_loader)
        
        
            ########################################################################
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)
            
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        # prep model for evaluation
    
    #     if step % checkpoint_freq == 0:
            results = {
                'step': step*steps_per_epoch*2,
                'epoch': step ,
                'cons_loss': contrastive_loss,
                    }
            for key, val in checkpoint_vals.items():
                mean_val = np.mean(val)
                results[key] = mean_val 

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
#         print(name)
                acc = misc.accuracy(local_model, loader, weights, device)
                results[name+'_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                                                colwidth=12)

            train_source_acc = np.mean([results[i] for i in train_source])
            print('train_source_acc', train_source_acc)
            test_source_acc = np.mean([results[i] for i in test_source])
            print('test_source_acc', test_source_acc)
        ######################################################################
        ######################### averging all weights ########################

            if not cli.merge(local_model):
                brake=True
                break

        
            ########################################################################
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)
            
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
    # prep model for evaluation
    
#     if step % checkpoint_freq == 0:
            results = {
                'step': step*steps_per_epoch*2,
                'epoch': step ,
                'cons_loss': contrastive_loss,
                    }
            for key, val in checkpoint_vals.items():
                mean_val = np.mean(val)
                results[key] = mean_val 

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
#         print(name)
                acc = misc.accuracy(local_model, loader, weights, device)
                results[name+'_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                                            colwidth=12)

            results.update({
                'hparams': hparams,
#                 'args': vars(args)    
                })


#     train_source_acc = np.mean([results[i] for i in train_source])
#     print('train_source_acc', train_source_acc)
#     test_source_acc = np.mean([results[i] for i in test_source])
#     print('test_source_acc', test_source_acc)
            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])
    
            if brake:
                break        
        
 
        # Rest of the client code
    except Exception as e:
        print(f"An error occurred: {e}")
        break

del cli
