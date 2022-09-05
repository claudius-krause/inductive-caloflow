# pylint: disable=invalid-name
""" Main script to run iterative flow for the CaloChallenge, datasets 2 and 3.

    by Claudius Krause, Matthew Buckley, Gopolang Mohlabeng, David Shih

"""

######################################   Imports   ################################################

import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from nflows import transforms, distributions, flows
import h5py

from data import get_calo_dataloader

torch.set_default_dtype(torch.float64)


#####################################   Parser setup   ############################################
parser = argparse.ArgumentParser()

parser.add_argument('--which_ds', default='2',
                    help='Which dataset to use: "2", "3" ')

# which flow uses bit flags: flow 1 counts 1, flow 2 counts 2, flow 3 counts 4.
# sum up which ones you want to work with
parser.add_argument('--which_flow', type=int, default=7,
                    help='Which flow(s) to train/evaluate/generate. Default 7(=1+2+3).')
parser.add_argument('--train', action='store_true', help='train the setup')
parser.add_argument('--generate', action='store_true',
                    help='generate from a trained flow and plot')
parser.add_argument('--evaluate', action='store_true', help='evaluate LL of a trained flow')

parser.add_argument('--student_mode', action='store_true',
                    help='Work with IAF-student instead of MAF-teacher')

parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--results_file', default='results.txt',
                    help='Filename where to store settings and test results.')
parser.add_argument('--restore_file', type=str, default=None, help='Model file to restore.')
parser.add_argument('--student_restore_file', type=str, default=None,
                    help='Student model file to restore.')
parser.add_argument('--data_dir', default='/home/claudius/ML_source/CaloChallenge/official',
                    help='Where to find the training dataset')



parser.add_argument('--noise_level', type=float, default=5e-3,
                    help='What level of noise to add to training data. Default is 5e-3')
parser.add_argument('--threshold_cut', type=float, default=1.5e-2,
                    help='What cut to apply after generation. Default is 1.5e-2')

# MAF parameters
parser.add_argument('--n_blocks', type=int, default='8',
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size of flow training. Defaults to 1000.')
parser.add_argument('--student_n_blocks', type=int, default=8,
                    help='Total number of blocks to stack in the student model (MADE in IAF).')
parser.add_argument('--hidden_size', type=int, default='256',
                    help='Hidden layer size for each MADE block in an MAF.')
parser.add_argument('--student_hidden_size', type=int, default=378,
                    help='Hidden layer size for each MADE block in the student IAF.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability, defaults to 0')

#normalization = {'2': 64172.594645065976, '3': 63606.50492698312} # or 6.5e4
#


#######################################   helper functions   ######################################

ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def sigmoid(x):
    """ returns sigmoid of input """
    return torch.exp(x) / (torch.exp(x) + 1.)

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp(clamp_low, clamp_high)

def add_noise(input_array, noise_level=1e-4):
    """ adds a bit of noise """
    noise = (torch.rand(size=input_array.size())*noise_level).to(input_array.device)
    return input_array+noise
    #return (input_array+noise)/(1.+noise_level)

def save_flow(model, number, arg):
    """ saves model to file """
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(arg.output_dir, 'ds_{}_flow_{}.pt'.format(arg.which_ds, number)))
    print("Model saved")

def load_flow(model, number, arg):
    """ loads model from file """
    checkpoint = torch.load(os.path.join(arg.output_dir,
                                         'ds_{}_flow_{}.pt'.format(arg.which_ds, number)),
                            map_location=arg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(arg.device)
    model.eval()
    return model

class IAFRQS(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform):
    """ IAF version of nflows MAF-RQS"""
    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)
    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

class GuidedCompositeTransform(transforms.CompositeTransform):
    """Composes several transforms into one (in the order they are given),
       optionally returns intermediate results (steps) and NN outputs (p)"""

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__(transforms)
        self._transforms = torch.nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, direction, return_steps=False, return_p=False):
        steps = [inputs]
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        ret_p = []
        for func in funcs:
            if hasattr(func.__self__, '_transform') and return_p:
                # in student IAF
                if direction == 'forward':
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                else:
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
            elif hasattr(func.__self__, 'autoregressive_net') and return_p:
                # in teacher MAF
                if direction == 'forward':
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
                else:
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
            else:
                outputs, logabsdet = func(outputs, context)
            steps.append(outputs)
            total_logabsdet += logabsdet
        if return_steps and return_p:
            return outputs, total_logabsdet, steps, ret_p
        elif return_steps:
            return outputs, total_logabsdet, steps
        elif return_p:
            return outputs, total_logabsdet, ret_p
        else:
            return outputs, total_logabsdet

    def forward(self, inputs, context=None, return_steps=False, return_p=False):
        #funcs = self._transforms
        funcs = (transform.forward for transform in self._transforms)
        return self._cascade(inputs, funcs, context, direction='forward',
                             return_steps=return_steps, return_p=return_p)

    def inverse(self, inputs, context=None, return_steps=False, return_p=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, direction='inverse',
                             return_steps=return_steps, return_p=return_p)

def build_flow(features, context_features, arg, hidden_size, num_layers=1):
    """ returns build flow and optimizer """
    flow_params_RQS = {'num_blocks': num_layers, # num of hidden layers per block
                       'use_residual_blocks': False,
                       'use_batch_norm': False,
                       'dropout_probability': arg.dropout_probability,
                       'activation': F.relu,
                       'random_mask':False,
                       'num_bins': arg.n_bins,
                       'tails':'linear',
                       'tail_bound': 14.,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(arg.n_blocks):
        flow_blocks.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            **flow_params_RQS,
            features=features,
            context_features=context_features,
            hidden_features=hidden_size))
        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(features))
        else:
            flow_blocks.append(transforms.RandomPermutation(features))

    del flow_blocks[-1]
    flow_transform = GuidedCompositeTransform(flow_blocks)
    flow_base_distribution = distributions.StandardNormal(shape=[features])

    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[400, 500], gamma=0.5,
                                                       verbose=True)
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Flow has {} parameters".format(total_parameters))
    print("Flow has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer, lr_schedule

def train_eval_flow_1(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 1, learning p(E_i|E_inc), eval after each epoch """

    num_epochs = 750
    best_LL = -np.inf
    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            e_dep = batch['energy_dep'].to(arg.device)
            e_dep = logit_trafo(e_dep/arg.normalization)
            cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5
            loss = - flow.log_prob(e_dep, cond).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                      file=open(arg.results_file, 'a'))

        logprb_mean, logprb_std = eval_flow_1(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 1, arg)
        schedule.step()
    flow = load_flow(flow, 1, arg)

@torch.no_grad()
def eval_flow_1(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 1"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        e_dep = batch['energy_dep'].to(arg.device)
        e_dep = logit_trafo(e_dep/arg.normalization)
        cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5

        loglike.append(flow.log_prob(e_dep, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std

@torch.no_grad()
def generate_flow_1(flow, arg, num_samples, energies=None):
    """ samples from flow 1 and returns E_i and E_inc in MeV """
    start_time = time.time()
    if energies is None:
        energies = (torch.rand(size=(num_samples, 1))*3. - 1.5).to(arg.device)
    samples = flow.sample(1, energies).reshape(len(energies), -1)
    samples = inverse_logit(samples) * arg.normalization
    samples = torch.where(samples < arg.threshold_cut, torch.zeros_like(samples), samples)
    end_time = time.time()
    total_time = end_time-start_time
    time_string = "Flow 1: Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, 1*len(energies),
                             1, total_time*1e3 / (1*len(energies))))
    print(time_string.format(int(total_time//60), total_time%60, 1*len(energies),
                             1, total_time*1e3 / (1*len(energies))),
          file=open(arg.results_file, 'a'))
    return 10**(energies + 4.5), samples

def train_eval_flow_2(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 2, learning p(I_0|E_0, E_inc) eval after each epoch"""

    num_epochs = 750
    best_LL = -np.inf
    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            shower = batch['layer'].to(arg.device)
            cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
            cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)/4.
            cond = torch.vstack([cond_inc.T, cond_dep.T]).T
            loss = - flow.log_prob(shower, cond).mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                      file=open(arg.results_file, 'a'))
        logprb_mean, logprb_std = eval_flow_2(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 2, arg)
        schedule.step()
    flow = load_flow(flow, 2, arg)

@torch.no_grad()
def eval_flow_2(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 2"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        shower = batch['layer'].to(arg.device)
        cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
        cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)/4.
        cond = torch.vstack([cond_inc.T, cond_dep.T]).T

        loglike.append(flow.log_prob(shower, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std

@torch.no_grad()
def generate_flow_2(flow, arg, incident_en, samp_1):
    """ samples from flow 2 and returns I_0 for given E_0 and E_inc in MeV """
    start_time = time.time()
    cond_inc = torch.log10(incident_en.to(arg.device))-4.5
    cond_dep = logit_trafo(samp_1[:, 0].to(arg.device)/arg.normalization)/4.
    cond = torch.vstack([cond_inc.T, cond_dep.T]).T

    samples = flow.sample(1, cond).reshape(len(cond), -1)
    samples = inverse_logit(samples)
    #samples = samples/samples.sum(dim=-1, keepdims=True)* samp_1[:, 0].reshape(-1, 1)
    samples = samples * samp_1[:, 0].reshape(-1, 1)
    samples = torch.where(samples < arg.threshold_cut, torch.zeros_like(samples), samples)
    end_time = time.time()
    total_time = end_time-start_time
    time_string = "Flow 2: Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, 1*len(incident_en),
                             1, total_time*1e3 / (1*len(incident_en))))
    print(time_string.format(int(total_time//60), total_time%60, 1*len(incident_en),
                             1, total_time*1e3 / (1*len(incident_en))),
          file=open(arg.results_file, 'a'))
    return samples

def train_eval_flow_3(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 3, learning p(I_n|I_(n-1), E_n, E_(n-1), E_inc) eval after each epoch"""

    num_epochs = 40 # (dataset is 44x larger)
    best_LL = -np.inf
    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            shower = batch['layer'].to(arg.device)
            cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
            cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)
            cond_dep_p = logit_trafo(batch['energy_dep_p'].to(arg.device)/arg.normalization)
            cond_p = batch['layer_p'].to(arg.device)
            cond_num = F.one_hot(batch['layer_number']-1, num_classes=44).to(arg.device)
            cond = torch.vstack([cond_inc.T, cond_dep.T, cond_dep_p.T, cond_p.T, cond_num.T]).T
            loss = - flow.log_prob(shower, cond).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 25 == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                      file=open(arg.results_file, 'a'))
            if (idx % 250 == 0) and (idx != 0): # since dataset is so large
                logprb_mean, logprb_std = eval_flow_3(test_loader, flow, arg)

                output = 'Intermediate evaluate (epoch {}) -- '.format(epoch+1) +\
                    'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
                print(output.format(logprb_mean, logprb_std))
                print(output.format(logprb_mean, logprb_std),
                      file=open(arg.results_file, 'a'))
                if logprb_mean > best_LL:
                    best_LL = logprb_mean
                    save_flow(flow, 3, arg)
                schedule.step()

        logprb_mean, logprb_std = eval_flow_3(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 3, arg)
    flow = load_flow(flow, 3, arg)

@torch.no_grad()
def eval_flow_3(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 3"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        shower = batch['layer'].to(arg.device)
        cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
        cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)
        cond_dep_p = logit_trafo(batch['energy_dep_p'].to(arg.device)/arg.normalization)
        cond_p = batch['layer_p'].to(arg.device)
        cond_num = F.one_hot(batch['layer_number']-1, num_classes=44).to(arg.device)
        cond = torch.vstack([cond_inc.T, cond_dep.T, cond_dep_p.T, cond_p.T, cond_num.T]).T
        loglike.append(flow.log_prob(shower, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std

@torch.no_grad()
def generate_flow_3(flow, arg, incident_en, samp_1, samp_2):
    """ samples from flow 2 and returns I_0 for given E_0 and E_inc in MeV """
    start_time = time.time()
    full_sample = [samp_2]
    cond_inc = torch.log10(incident_en.to(arg.device))-4.5
    for i in range(1, 45):
        cond_dep = logit_trafo(samp_1[:, i].to(arg.device)/arg.normalization)
        cond_dep_p = logit_trafo(samp_1[:, i-1].to(arg.device)/arg.normalization)
        cond_p = add_noise(full_sample[-1].to(arg.device), noise_level=arg.noise_level)
        cond_p = logit_trafo(cond_p / cond_p.sum(dim=-1, keepdims=True))
        cond_num = F.one_hot((i*torch.ones(size=(len(cond_dep), )).to(int))-1,
                             num_classes=44).to(arg.device)
        cond = torch.vstack([cond_inc.T, cond_dep.T, cond_dep_p.T, cond_p.T, cond_num.T]).T
        samples = flow.sample(1, cond).reshape(len(cond), -1)
        samples = inverse_logit(samples)
        #samples = samples/samples.sum(dim=-1, keepdims=True)* samp_1[:, i].reshape(-1, 1)
        samples = samples * samp_1[:, i].reshape(-1, 1)
        samples = torch.where(samples < arg.threshold_cut, torch.zeros_like(samples), samples)
        full_sample.append(samples)
        print("Done sampling Calolayer {}/44.".format(i))
    end_time = time.time()
    full_sample = torch.cat(full_sample, dim=-1)
    total_time = end_time-start_time
    time_string = "Flow 3: Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, 1*len(incident_en),
                             1, total_time*1e3 / (1*len(incident_en))))
    print(time_string.format(int(total_time//60), total_time%60, 1*len(incident_en),
                             1, total_time*1e3 / (1*len(incident_en))),
          file=open(arg.results_file, 'a'))
    return full_sample

def save_to_file(incident, shower, arg):
    """ saves incident energies and showers to hdf5 file """
    filename = os.path.join(arg.output_dir, 'inductive_ds_{}.hdf5'.format(arg.which_ds))
    dataset_file = h5py.File(filename, 'w')
    dataset_file.create_dataset('incident_energies',
                                data=incident.reshape(len(incident), -1), compression='gzip')
    dataset_file.create_dataset('showers',
                                data=shower.reshape(len(shower), -1), compression='gzip')
    dataset_file.close()

def renormalize_and_cut(raw_samps, target_en, thres):
    """ renormalizes raw samples, such that they sum to target energy
        sets all voxel below threshold to 0
        ensures that both conditions above hold simultaneously
        avoids for loops!
    """
    len_batch, num_voxel = raw_samps.size()
    # sort voxel by energy
    sorted_showers, sorted_showers_idx = torch.sort(raw_samps, dim=-1)

    # that is equivalent to a reverse cumsum. element i is given by (sum elements >= i)
    # in the context here this means: index i is the energy of the layer if the i lowest voxel
    # are set to 0
    summed_showers = sorted_showers + torch.sum(sorted_showers, dim=-1, keepdims=True) -\
        torch.cumsum(sorted_showers, dim=-1)

    # renormalization candidates. target E_i divided by the layer energy from above
    renorm_cand = target_en.view(-1, 1) / summed_showers

    # assume candidate and compute value of non-zero minimum after cut and renorm.
    renorm_showers_pivot = sorted_showers * renorm_cand

    # check which of those are above the desired cut, and find their index.
    # set index to max if below cut.
    # renorm to be used is then min of that
    selected_idx = torch.where(
        torch.gt(renorm_showers_pivot, thres),
        torch.arange(num_voxel).repeat(len_batch, 1),
        (num_voxel*torch.ones((len_batch, num_voxel))).long()).min(dim=-1, keepdims=True)[0]
    zero_mask = torch.where((torch.arange(num_voxel).repeat(len_batch, 1) < selected_idx),
                            torch.zeros_like(raw_samps), torch.ones_like(raw_samps))
    zero_mask = torch.gather(zero_mask, -1, torch.argsort(sorted_showers_idx))

    # now get the renormalization factor
    renorm = torch.gather(torch.cat((renorm_cand, torch.zeros(len_batch, 1)), dim=-1), -1,
                          selected_idx)

    # do the renormalization
    renormed_showers = raw_samps * renorm

    # cut away the too dim voxel (threshold=cut does not work!)
    cut_renormed_showers = renormed_showers * zero_mask

    return cut_renormed_showers


###################################################################################################
#######################################   running the code   ######################################
###################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    LAYER_SIZE = {'2': 9 * 16, '3': 18 * 50}[args.which_ds]
    DEPTH = 45
    args.normalization = 6.5e4

    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))

    preprocessing_kwargs = {'with_noise': True,
                            'noise_level': args.noise_level,
                            'apply_logit': True,
                            'do_normalization': True}

    if bin(args.which_flow)[-1] == '1':
        print("Working on Flow 1")
        print("Working on Flow 1", file=open(args.results_file, 'a'))
        if args.train or args.evaluate:
            train_loader_1, test_loader_1 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                1, args.device,
                which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

        flow_1, optimizer_1, schedule_1 = build_flow(DEPTH, 1, args, args.hidden_size)

        if args.train:
            train_eval_flow_1(flow_1, optimizer_1, schedule_1, train_loader_1, test_loader_1, args)

        if args.evaluate:
            flow_1 = load_flow(flow_1, 1, args)
            logprob_mean, logprob_std = eval_flow_1(test_loader_1, flow_1, args)
            output = 'Evaluate (flow 1) -- ' +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std),
                  file=open(args.results_file, 'a'))

        if args.generate:
            num_events = 10000
            num_batches = 10
            flow_1 = load_flow(flow_1, 1, args)
            incident_energies = []
            samples_1 = []
            for gen_batch in range(num_batches):
                incident_energies_loc, samples_1_loc = generate_flow_1(flow_1, args, num_events)
                np.save(os.path.join(args.output_dir, 'e_inc_1_{}.npy'.format(gen_batch)),
                        incident_energies_loc.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_1_{}.npy'.format(gen_batch)),
                        samples_1_loc.cpu().numpy())
                incident_energies.append(incident_energies_loc.cpu().numpy())
                samples_1.append(samples_1_loc.cpu().numpy())
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches))
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches),
                      file=open(args.results_file, 'a'))
            incident_energies = np.concatenate([*incident_energies])
            samples_1 = np.concatenate([*samples_1])
            np.save(os.path.join(args.output_dir, 'e_inc_1.npy'), incident_energies)
            np.save(os.path.join(args.output_dir, 'samples_1.npy'), samples_1)

    if bin(args.which_flow)[-2] == '1':
        print("Working on Flow 2")
        print("Working on Flow 2", file=open(args.results_file, 'a'))

        if args.train or args.evaluate:
            train_loader_2, test_loader_2 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                2, args.device,
                which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

        flow_2, optimizer_2, schedule_2 = build_flow(LAYER_SIZE, 2, args, args.hidden_size,
                                                     num_layers=2)

        if args.train:
            train_eval_flow_2(flow_2, optimizer_2, schedule_2, train_loader_2, test_loader_2, args)

        if args.evaluate:
            flow_2 = load_flow(flow_2, 2, args)
            logprob_mean, logprob_std = eval_flow_2(test_loader_2, flow_2, args)
            output = 'Evaluate (flow 2) -- ' +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std),
                  file=open(args.results_file, 'a'))

        if args.generate:
            if args.which_ds == '2':
                num_events = 10000
                num_batches = 10
            else:
                num_events = 5000
                num_batches = 20
            flow_1, _, _ = build_flow(DEPTH, 1, args, args.hidden_size)
            flow_1 = load_flow(flow_1, 1, args)
            flow_2 = load_flow(flow_2, 2, args)
            incident_energies = []
            samples_1 = []
            samples_2 = []
            for gen_batch in range(num_batches):
                incident_energies_loc, samples_1_loc = generate_flow_1(flow_1, args, num_events)
                np.save(os.path.join(args.output_dir, 'e_inc_1_{}.npy'.format(gen_batch)),
                        incident_energies_loc.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_1_{}.npy'.format(gen_batch)),
                        samples_1_loc.cpu().numpy())
                incident_energies.append(incident_energies_loc.cpu().numpy())
                samples_1.append(samples_1_loc.cpu().numpy())
                samples_2_loc = generate_flow_2(flow_2, args, incident_energies_loc, samples_1_loc)
                np.save(os.path.join(args.output_dir, 'samples_2_{}.npy'.format(gen_batch)),
                        samples_2_loc.cpu().numpy())
                samples_2.append(samples_2_loc.cpu().numpy())
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches))
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches),
                      file=open(args.results_file, 'a'))
            incident_energies = np.concatenate([*incident_energies])
            samples_1 = np.concatenate([*samples_1])
            samples_2 = np.concatenate([*samples_2])
            np.save(os.path.join(args.output_dir, 'e_inc_1.npy'), incident_energies)
            np.save(os.path.join(args.output_dir, 'samples_1.npy'), samples_1)
            np.save(os.path.join(args.output_dir, 'samples_2.npy'), samples_2)

    if bin(args.which_flow)[-3] == '1':
        print("Working on Flow 3")
        print("Working on Flow 3", file=open(args.results_file, 'a'))

        if args.train or args.evaluate:
            train_loader_3, test_loader_3 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                3, args.device,
                which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

        flow_3, optimizer_3, schedule_3 = build_flow(LAYER_SIZE, 3+LAYER_SIZE+44, args,
                                                     args.hidden_size, num_layers=2)

        if args.train:
            train_eval_flow_3(flow_3, optimizer_3, schedule_3, train_loader_3, test_loader_3, args)

        if args.evaluate:
            flow_3 = load_flow(flow_3, 3, args)
            logprob_mean, logprob_std = eval_flow_3(test_loader_3, flow_3, args)
            output = 'Evaluate (flow 3) -- ' +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std),
                  file=open(args.results_file, 'a'))

        if args.generate:
            num_events = 10000
            num_batches = 10
            full_start_time = time.time()
            flow_1, _, _ = build_flow(DEPTH, 1, args, args.hidden_size)
            flow_1 = load_flow(flow_1, 1, args)
            flow_2, _, _ = build_flow(LAYER_SIZE, 2, args, args.hidden_size, num_layers=2)
            flow_2 = load_flow(flow_2, 2, args)
            flow_3 = load_flow(flow_3, 3, args)
            incident_energies = []
            samples_1 = []
            samples_2 = []
            samples_3 = []
            for gen_batch in range(num_batches):
                incident_energies_loc, samples_1_loc = generate_flow_1(flow_1, args, num_events)
                np.save(os.path.join(args.output_dir, 'e_inc_1_{}.npy'.format(gen_batch)),
                        incident_energies_loc.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_1_{}.npy'.format(gen_batch)),
                        samples_1_loc.cpu().numpy())
                incident_energies.append(incident_energies_loc.cpu().numpy())
                samples_1.append(samples_1_loc.cpu().numpy())
                samples_2_loc = generate_flow_2(flow_2, args, incident_energies_loc, samples_1_loc)
                np.save(os.path.join(args.output_dir, 'samples_2_{}.npy'.format(gen_batch)),
                        samples_2_loc.cpu().numpy())
                samples_2.append(samples_2_loc.cpu().numpy())
                samples_3_loc = generate_flow_3(flow_3, args, incident_energies_loc,
                                                samples_1_loc, samples_2_loc)
                np.save(os.path.join(args.output_dir, 'samples_3_{}.npy'.format(gen_batch)),
                        samples_3_loc.cpu().numpy())
                samples_3.append(samples_3_loc.cpu().numpy())
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches))
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches),
                      file=open(args.results_file, 'a'))
            incident_energies = np.concatenate([*incident_energies])
            samples_1 = np.concatenate([*samples_1])
            samples_2 = np.concatenate([*samples_2])
            samples_3 = np.concatenate([*samples_3])
            full_end_time = time.time()
            save_to_file(incident_energies, samples_3, args)
            np.save(os.path.join(args.output_dir, 'e_inc_1.npy'), incident_energies)
            np.save(os.path.join(args.output_dir, 'samples_1.npy'), samples_1)
            np.save(os.path.join(args.output_dir, 'samples_2.npy'), samples_2)

            full_total_time = full_end_time - full_start_time
            time_str = "Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
                " This means {:.2f} ms per event."
            print(time_str.format(int(full_total_time//60), full_total_time%60,
                                  num_events*num_batches,
                                  num_batches, full_total_time*1e3 / (num_events*num_batches)))
            print(time_str.format(int(full_total_time//60), full_total_time%60, num_events,
                                  num_batches*num_batches,
                                  full_total_time*1e3 / (num_events*num_batches)),
                  file=open(args.results_file, 'a'))



    print("DONE with everything!")
    print("DONE with everything!", file=open(args.results_file, 'a'))
