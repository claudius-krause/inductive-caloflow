# pylint: disable=invalid-name
""" Main script to run iterative flow for the CaloChallenge, datasets 2 and 3.

    by Matthew Buckley, Claudius Krause, Ian Pang and David Shih

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
from nflows.utils import torchutils
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
parser.add_argument('--data_dir', default='./data_dir',
                    help='Where to find the training dataset')

parser.add_argument('--log_interval', type=int, default=175,
                    help='How often to show loss statistics and save samples.')

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
parser.add_argument('--student_hidden_size', type=int, default='256',
                    help='Hidden layer size for each MADE block in the student IAF.')
parser.add_argument('--student_width', type=float, default=1.,
                    help='Width of the base dist. that is used for student training.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability, defaults to 0')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Sets the relative weight between z-chi2 loss (beta=0) and x-chi2 loss')
#normalization = {'2': 64172.594645065976, '3': 63606.50492698312} # or 6.5e4

#######################################   helper functions   ######################################

ALPHA = 1e-6 
#ALPHA = 1e-1
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

def save_flow(model, number, arg):
    """ saves model to file """
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(arg.output_dir, 'ds_{}_flow_{}.pt'.format(arg.which_ds, number)))
    print("Model saved")

def save_flow_student(model, number, arg):
    """ saves model to file """
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(arg.output_dir, 'ds_{}_flow_{}_student.pt'.format(arg.which_ds, number)))
    print("Student model saved")

def load_flow(model, number, arg):
    """ loads model from file """
    checkpoint = torch.load(os.path.join(arg.output_dir,
                                         'ds_{}_flow_{}.pt'.format(arg.which_ds, number)),
                            map_location=arg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(arg.device)
    model.eval()
    return model

def load_flow_student(model, number, arg):
    """ loads model from file """
    checkpoint = torch.load(os.path.join(arg.output_dir,
                                         'ds_{}_flow_{}_student.pt'.format(arg.which_ds, number)),
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

def chi2_loss(input1, input2):
    ret = (((input1 - input2)**2).sum(dim=1)).mean()
    return ret

def logabsdet_of_base(noise, width=1.):
    """ for computing KL of student"""
    shape = noise.size()[1]
    ret = -0.5 * torchutils.sum_except_batch((noise/width) ** 2, num_batch_dims=1)
    log_z = torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64)
    return ret - log_z

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

    if arg.which_ds == '3':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[400, 500], gamma=0.5,
                                                       verbose=True)
    elif context_features == 1 and arg.which_ds == '2':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1E-4, total_steps=35000, epochs=500, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False)
    elif context_features == 2 and arg.which_ds == '2':
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1E-3, total_steps=14000, epochs=200, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=50.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False) 
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1E-3, total_steps=184800, epochs=60, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=50.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False) 
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Flow has {} parameters".format(total_parameters))
    print("Flow has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer, lr_schedule

def build_flow_student(teacher, features, context_features, arg, student_hidden_size, num_layers=1):
    """ returns build student flow and student optimizer """
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

    teacher_perm = []

    for elem in teacher._transform._transforms:
        if hasattr(elem, '_permutation'):
            teacher_perm.append(elem._permutation.to('cpu'))

    teacher_perm.append(teacher_perm[-1])
    flow_blocks = []
    #student_perms = []
    for i in range(arg.n_blocks):
        flow_blocks.append(
            transforms.InverseTransform(
                IAFRQS(
                    **flow_params_RQS,
                    features=features,
                    context_features=context_features,
                    hidden_features=student_hidden_size
                )))

        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(features))
        else:
            flow_blocks.append(transforms.Permutation(teacher_perm[i]))
    del flow_blocks[-1]


    flow_transform = GuidedCompositeTransform(flow_blocks)
    
    flow_base_distribution = distributions.StandardNormal(shape=[features])
    student = flows.Flow(transform=flow_transform,
                            distribution=flow_base_distribution).to(arg.device)

    print(student)
    print(student, file=open(arg.results_file, 'a'))

    total_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print("Student has {} parameters".format(int(total_parameters)))
    print("Student has {} parameters".format(int(total_parameters)),
            file=open(arg.results_file, 'a'))

    if context_features == 2 and arg.which_ds == '2':
        optimizer_student = torch.optim.Adam(student.parameters(), lr=2e-5)
        lr_schedule_student = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=1E-3, total_steps=28000, epochs=400, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=50.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False) 
    elif context_features == 2 and arg.which_ds == '3':
        optimizer_student = torch.optim.Adam(student.parameters(), lr=2e-5)   
        lr_schedule_student = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=1E-3, total_steps=280000, epochs=400, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=50.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False)  
    elif context_features == 191:
        optimizer_student = torch.optim.Adam(student.parameters(), lr=2e-5)   
        lr_schedule_student = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=5E-4, total_steps=308000, epochs=100, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.80, max_momentum=0.95, div_factor=25.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False)
    else:  
        optimizer_student = torch.optim.Adam(student.parameters(), lr=4e-6)   
        lr_schedule_student = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=1E-4, total_steps=528000, epochs=30, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.80, max_momentum=0.95, div_factor=25.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False) 
    return student, optimizer_student, lr_schedule_student

############################################### Flow 1 ###########################################################

def train_eval_flow_1(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 1, learning p(E_i|E_inc), eval after each epoch """

    if arg.which_ds == '2': 
        num_epochs = 500 
    else: 
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
            if arg.which_ds == '2':  schedule.step()
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
        if arg.which_ds == '3': schedule.step()
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

################################################### Flow 2 ####################################################

def train_eval_flow_2(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 2 teacher, learning p(I_0|E_0, E_inc) eval after each epoch"""

    if arg.which_ds == '2': 
        num_epochs = 200
    else:
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
            schedule.step()
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
        #schedule.step()
    flow = load_flow(flow, 2, arg)

def train_eval_flow_2_student(teacher, student, student_schedule, train_loader, test_loader, optimizer_student, features, arg):
    """ train flow 2 student, learning p(I_0|E_0, E_inc) eval after each epoch"""
    best_eval_KL = float('inf')
    num_epochs = 400
    
    for i in range(num_epochs):
        optimizer_student.zero_grad()
        eval_KL = train_flow_2_student(teacher, student, train_loader, optimizer_student, student_schedule, i, features, arg)
        #args.test_loss.append(eval_KL)
        if eval_KL < best_eval_KL:
            best_eval_KL = eval_KL
            save_flow_student(student, 2, arg)

def train_flow_2_student(teacher, student, dataloader, optimizer, student_schedule, epoch, features, arg):
    """ train flow 2 student with recursive single teacher """
    teacher.eval()
    student.train()

    KL = []
    for idx, batch in enumerate(dataloader):
        #shower = batch['layer'].to(arg.device)
        cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
        cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)/4.
        cond = torch.vstack([cond_inc.T, cond_dep.T]).T

        z_cond = cond

        x = batch['layer'].to(arg.device)

        # data MSE (x-space)
        teacher_noise, _, data_steps_teacher, data_p_teacher = \
                    teacher._transform.forward(x, cond, return_steps=True, return_p=True)
        student_data, _, data_steps_student, data_p_student = \
                    student._transform.inverse(teacher_noise, cond, return_steps=True, return_p=True)
                
        loss_chi_x = chi2_loss(student_data, x)
        for i, step in enumerate(data_steps_student):
            loss_chi_x += chi2_loss(step, data_steps_teacher[-i-1])
        for i, step in enumerate(data_p_student):
            loss_chi_x += chi2_loss(step, data_p_teacher[-i-1])
        # latent MSE (z-space):
        noise = (torch.randn(args.batch_size, features)*args.student_width).to(args.device)
        pts, log_student_pre, latent_steps_student, latent_p_student = \
            student._transform.inverse(noise, z_cond, return_steps=True, return_p=True)
        latent_teacher_noise, log_teacher_pre, latent_steps_teacher, latent_p_teacher = \
            teacher._transform.forward(pts, z_cond, return_steps=True, return_p=True)
        loss_chi_z = chi2_loss(latent_teacher_noise, noise)
        for i, step in enumerate(latent_steps_student):
            loss_chi_z += chi2_loss(step, latent_steps_teacher[-i-1])
        for i, step in enumerate(latent_p_student):
            loss_chi_z += chi2_loss(step, latent_p_teacher[-i-1])
        with torch.no_grad():
            # KL:
            logabsdet_noise_student = logabsdet_of_base(noise, width=args.student_width)
            logabsdet_noise_teacher = logabsdet_of_base(latent_teacher_noise,
                                                        width=args.student_width)

            log_teacher = logabsdet_noise_teacher + log_teacher_pre
            log_student = logabsdet_noise_student - log_student_pre

            KL_local = (log_student-log_teacher.detach()).mean()
            KL.append(log_student-log_teacher.detach())
        loss = (1.-args.beta) * loss_chi_z + args.beta * loss_chi_x

        if epoch < 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            student_schedule.step()
    
        if idx % 10 == 0:
            print_string = 'epoch {:3d} / {}, step {:4d} / {}; x-chi {:.4f}; '+\
            'z-chi {:.4f}; loss {:.4f}; KL {:.4f} '
            print(print_string.format(
                epoch+1, 400, idx, len(dataloader), loss_chi_x.item(), loss_chi_z.item(),
                loss.item(), KL_local.item()))
            print(print_string.format(
                epoch+1, 400, idx, len(dataloader), loss_chi_x.item(), loss_chi_z.item(),
                loss.item(), KL_local.item()), file=open(args.results_file, 'a'))
    KL_numpy = torch.cat(KL, dim=0).to('cpu').numpy()
    KL_mean = KL_numpy.mean()
    KL_std = KL_numpy.std()
    print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))))
    print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))),
          file=open(args.results_file, 'a'))
    del loss
    return KL_mean

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

######################################### Flow 3 ################################################

def train_eval_flow_3(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 3, learning p(I_n|I_(n-1), E_n, E_(n-1), E_inc) eval after each epoch"""

    if arg.which_ds == '3':
        num_epochs = 20 
    else:
        num_epochs = 60 
    if vars(arg).get('best_LL') is None:
        best_LL = -np.inf
    else:
        best_LL = arg.best_LL
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
            if arg.which_ds == '2': schedule.step()
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
                if arg.which_ds == '3': schedule.step()

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
    arg.best_LL = best_LL

def train_eval_flow_3_student(teacher, student, student_schedule, train_loader, test_loader, optimizer_student, features, arg):
    """ train flow 3 student, learning p(I_n|I_(n-1), E_n, E_(n-1), E_inc) eval after each epoch"""
    best_eval_KL = float('inf')
    if arg.which_ds == '3':
        num_epochs = 15
    else:
        num_epochs = 100
    
    for i in range(num_epochs):
        optimizer_student.zero_grad()
        teacher.eval()
        student.train()

        KL = []
        for idx, batch in enumerate(train_loader):

            #shower = batch['layer'].to(arg.device)
            cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
            cond_dep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)
            cond_dep_p = logit_trafo(batch['energy_dep_p'].to(arg.device)/arg.normalization)
            cond_p = batch['layer_p'].to(arg.device)
            cond_num = F.one_hot(batch['layer_number']-1, num_classes=44).to(arg.device)
            cond = torch.vstack([cond_inc.T, cond_dep.T, cond_dep_p.T, cond_p.T, cond_num.T]).T

            z_cond = cond
            x = batch['layer'].to(arg.device)
            # data MSE (x-space)
            teacher_noise, _, data_steps_teacher, data_p_teacher = \
                        teacher._transform.forward(x, cond, return_steps=True, return_p=True)
            student_data, _, data_steps_student, data_p_student = \
                        student._transform.inverse(teacher_noise, cond, return_steps=True, return_p=True)
                    
            loss_chi_x = chi2_loss(student_data, x)
            for index, step in enumerate(data_steps_student):
                loss_chi_x += chi2_loss(step, data_steps_teacher[-index-1])
            for index, step in enumerate(data_p_student):
                loss_chi_x += chi2_loss(step, data_p_teacher[-index-1])

            # latent MSE (z-space):
            noise = (torch.randn(args.batch_size,features)).to(args.device)
            pts, log_student_pre, latent_steps_student, latent_p_student = \
                student._transform.inverse(noise, z_cond, return_steps=True, return_p=True)
            latent_teacher_noise, log_teacher_pre, latent_steps_teacher, latent_p_teacher = \
                teacher._transform.forward(pts, z_cond, return_steps=True, return_p=True)
            loss_chi_z = chi2_loss(latent_teacher_noise, noise)
            for index, step in enumerate(latent_steps_student):
                loss_chi_z += chi2_loss(step, latent_steps_teacher[-index-1])

            for index, step in enumerate(latent_p_student):
                loss_chi_z += chi2_loss(step, latent_p_teacher[-index-1])
            with torch.no_grad():
                # KL:
                logabsdet_noise_student = logabsdet_of_base(noise, width=args.student_width)
                logabsdet_noise_teacher = logabsdet_of_base(latent_teacher_noise,
                                                            width=args.student_width)

                log_teacher = logabsdet_noise_teacher + log_teacher_pre
                log_student = logabsdet_noise_student - log_student_pre

                KL_local = (log_student-log_teacher.detach()).mean()
                KL.append(log_student-log_teacher.detach())
            loss = (1.-args.beta) * loss_chi_z + args.beta * loss_chi_x
            
            if i < 0:
                optimizer_student.zero_grad()
                loss.backward()
                optimizer_student.step()
            else:
                loss.backward()
                optimizer_student.step()
                optimizer_student.zero_grad()
                student_schedule.step()
            
            if idx % 25 == 0:
                print_string = 'epoch {:3d} / {}, step {:4d} / {}; x-chi {:.4f}; '+\
                'z-chi {:.4f}; loss {:.4f}; KL {:.4f} '
                print(print_string.format(
                    i+1, num_epochs, idx, len(train_loader), loss_chi_x.item(), loss_chi_z.item(),
                    loss.item(), KL_local.item()))
                print(print_string.format(
                    i+1, num_epochs, idx, len(train_loader), loss_chi_x.item(), loss_chi_z.item(),
                    loss.item(), KL_local.item()), file=open(args.results_file, 'a'))

            if (idx % 250 == 0) and (idx != 0): # since dataset is so large
                KL_numpy = torch.cat(KL, dim=0).to('cpu').numpy()
                KL_mean = KL_numpy.mean()
                KL_std = KL_numpy.std()

                print("Intermediate KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))))
                print("Intermediate KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))),
            file=open(args.results_file, 'a'))
                if KL_mean < best_eval_KL:
                    best_eval_KL = KL_mean
                    save_flow_student(student, 3, arg)

        KL_numpy = torch.cat(KL, dim=0).to('cpu').numpy()
        KL_mean = KL_numpy.mean()
        KL_std = KL_numpy.std()
        print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))))
        print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))),
            file=open(args.results_file, 'a'))
        if KL_mean < best_eval_KL:
            best_eval_KL = KL_mean
            save_flow_student(student, 3, arg)
        #student_schedule.step()
        del loss
        #return KL_mean

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
    filename = os.path.join(arg.output_dir, 'inductive_ds_{}_challenge_test.hdf5'.format(arg.which_ds))
    dataset_file = h5py.File(filename, 'w')
    dataset_file.create_dataset('incident_energies',
                                data=incident.reshape(len(incident), -1), compression='gzip')
    dataset_file.create_dataset('showers',
                                data=shower.reshape(len(shower), -1), compression='gzip')
    dataset_file.close()


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

######################## Working with Flow 1 ######################################
    if bin(args.which_flow)[-1] == '1':
        print("Working on Flow 1")
        print("Working on Flow 1", file=open(args.results_file, 'a'))
        if args.train or args.evaluate:
            if args.which_ds == '2':
                train_loader_1, test_loader_1 = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)), 1, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs) 

            else: train_loader_1, test_loader_1 = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1+2.hdf5'.format(args.which_ds)), 1, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs) 

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

    ################################## Working with Teacher for Flows 2 and 3 ################################################
    if not args.student_mode:
        if bin(args.which_flow)[-2] == '1':
            print("Working on Flow 2 teacher")
            print("Working on Flow 2 teacher", file=open(args.results_file, 'a'))

            if args.train or args.evaluate:
                if args.which_ds == '2':
                    train_loader_2, test_loader_2 = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)), 2, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)
                else: train_loader_2, test_loader_2 = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1+2.hdf5'.format(args.which_ds)), 2, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

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
                    num_batches = 10
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
            print("Working on Flow 3 teacher")
            print("Working on Flow 3 teacher", file=open(args.results_file, 'a'))

            if args.train or args.evaluate:
                train_loader_3, test_loader_3 = get_calo_dataloader(
                    os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                    3, args.device,
                    which_ds=args.which_ds, batch_size=args.batch_size,
                    small_file=(args.which_ds == '3'), **preprocessing_kwargs)

            flow_3, optimizer_3, schedule_3 = build_flow(LAYER_SIZE, 3+LAYER_SIZE+44, args,
                                                        args.hidden_size,
                                                        num_layers=4-int(args.which_ds))

            if args.train:
                train_eval_flow_3(flow_3, optimizer_3, schedule_3, train_loader_3, test_loader_3, args)
                if args.which_ds == '3':
                    # train dataset 3 in two turns, with 2 source files
                    del train_loader_3, test_loader_3
                    train_loader_3, test_loader_3 = get_calo_dataloader(
                        os.path.join(args.data_dir, 'dataset_{}_2.hdf5'.format(args.which_ds)),
                        3, args.device, small_file=(args.which_ds == '3'),
                        which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)
                    train_eval_flow_3(flow_3, optimizer_3, schedule_3, train_loader_3, test_loader_3,
                                    args)

            if args.evaluate:
                flow_3 = load_flow(flow_3, 3, args)
                logprob_mean, logprob_std = eval_flow_3(test_loader_3, flow_3, args)
                output = 'Evaluate (flow 3) -- ' +\
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
    else:
        # here we are working with student models
        if bin(args.which_flow)[-2] == '1':
            print("Working on Flow 2 student")
            print("Working on Flow 2 student", file=open(args.results_file, 'a'))

            if args.train or args.evaluate:
                if args.which_ds == '2':
                    train_loader_2_student, test_loader_2_student = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)), 2, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)
                else: train_loader_2_student, test_loader_2_student = get_calo_dataloader(os.path.join(args.data_dir, 'dataset_{}_1+2.hdf5'.format(args.which_ds)), 2, args.device, which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

            flow_2, optimizer_2, schedule_2 = build_flow(LAYER_SIZE, 2, args, args.hidden_size,
                                                        num_layers=2)
            if args.train: flow_2 = load_flow(flow_2, 2, args)
            flow_2_student, optimizer_2_student, schedule_2_student = build_flow_student(flow_2, LAYER_SIZE, 2, args, 256,
                                                        num_layers=2)
            #flow_2 = load_flow(flow_2, 2, args)
            if args.train:
                train_eval_flow_2_student(flow_2, flow_2_student, schedule_2_student, train_loader_2_student, test_loader_2_student, optimizer_2_student, LAYER_SIZE, args)
            if args.evaluate:
                flow_2_student = load_flow(flow_2_student, 2, args)
                logprob_mean, logprob_std = eval_flow_2(test_loader_2_student, flow_2_student, args)
                output = 'Evaluate (flow 2 student) -- ' +\
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
                flow_2_student = load_flow_student(flow_2_student, 2, args)
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
                    samples_2_loc = generate_flow_2(flow_2_student, args, incident_energies_loc, samples_1_loc)
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

        ############################# Flow 3 student ###################
        if bin(args.which_flow)[-3] == '1':
            print("Working on Flow 3 student")
            print("Working on Flow 3 student", file=open(args.results_file, 'a'))

            if args.train or args.evaluate:
                train_loader_3_student, test_loader_3_student = get_calo_dataloader(
                    os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                    3, args.device,
                    which_ds=args.which_ds, batch_size=args.batch_size,
                    small_file=(args.which_ds == '3'), **preprocessing_kwargs)

            flow_3, optimizer_3, schedule_3 = build_flow(LAYER_SIZE, 3+LAYER_SIZE+44, args,
                                                        args.hidden_size,
                                                        num_layers=4-int(args.which_ds))
                                                        
            if args.train: flow_3 = load_flow(flow_3, 3, args)
            flow_3_student, optimizer_3_student, schedule_3_student = build_flow_student(flow_3, LAYER_SIZE, 3+LAYER_SIZE+44, args,
                                                        args.student_hidden_size, num_layers=4-int(args.which_ds))
            if args.train:
                train_eval_flow_3_student(flow_3, flow_3_student, schedule_3_student, train_loader_3_student, test_loader_3_student, optimizer_3_student, LAYER_SIZE, args)
                if args.which_ds == '3':
                    # train dataset 3 in two turns, with 2 source files                                                                                                                                                                                                       
                    del train_loader_3_student, test_loader_3_student
                    train_loader_3_student, test_loader_3_student = get_calo_dataloader(
                        os.path.join(args.data_dir, 'dataset_{}_2.hdf5'.format(args.which_ds)),
                        3, args.device, small_file=(args.which_ds == '3'),
                        which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)
                    train_eval_flow_3_student(flow_3, flow_3_student, schedule_3_student, train_loader_3_student, test_loader_3_student, optimizer_3_student, LAYER_SIZE, args)

            if args.evaluate:
                flow_3_student = load_flow(flow_3_student, 3, args)
                logprob_mean, logprob_std= eval_flow_3(test_loader_3_student, flow_3_student, args)
                output = 'Evaluate (flow 3 student) -- ' +\
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
                    num_batches = 10
                full_start_time = time.time()
                flow_1, _, _ = build_flow(DEPTH, 1, args, args.hidden_size)
                flow_1 = load_flow(flow_1, 1, args)
                flow_2, _ , _ = build_flow(LAYER_SIZE, 2, args, args.hidden_size,
                                                        num_layers=2)
                #flow_2 = load_flow(flow_2, 2, args)

                flow_2_student, _, _ = build_flow_student(flow_2, LAYER_SIZE, 2, args, 256,                                                                                                                                                             
                                                        num_layers=2)
                flow_2_student = load_flow_student(flow_2_student, 2, args)
                flow_3_student = load_flow_student(flow_3_student, 3, args)
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
                    samples_2_loc = generate_flow_2(flow_2_student, args, incident_energies_loc, samples_1_loc)                                                                                                                                     
                    np.save(os.path.join(args.output_dir, 'samples_2_{}.npy'.format(gen_batch)),
                            samples_2_loc.cpu().numpy())
                    samples_2.append(samples_2_loc.cpu().numpy())
                    samples_3_loc = generate_flow_3(flow_3_student, args, incident_energies_loc,
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
