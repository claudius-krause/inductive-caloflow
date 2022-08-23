# pylint: disable=invalid-name
""" Main script to run iterative flow for the CaloChallenge, datasets 2 and 3.

    by Claudius Krause, Matthew Buckley, Gopolang Mohlabeng, David Shih

"""

######################################   Imports   ################################################

import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from nflows import transforms, distributions, flows

from data import get_calo_dataloader

torch.set_default_dtype(torch.float64)


#####################################   Parser setup   ############################################
parser = argparse.ArgumentParser()

parser.add_argument('--which_ds', default='2',
                    help='Which dataset to use: "2", "3" ')

# todo: set which (subset) of the three flows tow work with/on
parser.add_argument('--which_flow', type=int, default=3,
                    help='Up to which flow to train/evaluate/generate. Default 3.')
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



parser.add_argument('--noise_level', default=1e-4,
                    help='What level of noise to add to training data. Default is 1e-4')

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

def build_flow(features, context_features, arg):
    """ returns build flow and optimizer """
    flow_params_RQS = {'num_blocks': 1, # num of hidden layers per block
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
            hidden_features=arg.hidden_size))
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
    #lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 900], gamma=0.5, verbose=True)
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Flow has {} parameters".format(total_parameters))
    print("Flow has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer

def train_eval_flow_1(flow, optimizer, train_loader, test_loader, arg):
    """ train flow 1, learning p(E_i|E_inc), eval after each epoch """

    num_epochs = 1000
    best_LL = -np.inf
    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            e_dep = batch['energy_dep'].to(arg.device)
            e_dep = logit_trafo(e_dep/6.5e4)
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
        # evaluate:
        #loglike = []
        #for idx, batch in enumerate(test_loader):
        #    flow.eval()
        #    e_dep = batch['energy_dep'].to(arg.device)
        #    e_dep = logit_trafo(e_dep/6.5e4)
        #    cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5

        #    with torch.no_grad():
        #        loglike.append(flow.log_prob(e_dep, cond))

        #logprobs = torch.cat(loglike, dim=0)

        #logprob_mean = logprobs.mean(0)
        #logprob_std = logprobs.var(0).sqrt()

        logprob_mean, logprob_std = eval_flow_1(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprob_mean, logprob_std))
        print(output.format(logprob_mean, logprob_std),
              file=open(arg.results_file, 'a'))
        if logprob_mean > best_LL:
            best_LL = logprob_mean
            #torch.save({'model_state_dict': flow.state_dict()},
            #           os.path.join(arg.output_dir, 'ds_{}_flow_1.pt'.format(arg.which_ds)))
            #print("Model saved")
            save_flow(flow, 1, arg)
    # load func:
    #checkpoint = torch.load(os.path.join(args.output_dir, 'ds_{}_flow_1.pt'.format(arg.which_ds)),
    #                        map_location=arg.device)
    #flow.load_state_dict(checkpoint['model_state_dict'])
    #flow.to(arg.device)
    #flow.eval()
    flow = load_flow(flow, 1, arg)

def eval_flow_1(test_loader, flow, arg):
    """ returns LL of data in dataloader """
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        e_dep = batch['energy_dep'].to(arg.device)
        e_dep = logit_trafo(e_dep/6.5e4)
        cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5

        with torch.no_grad():
            loglike.append(flow.log_prob(e_dep, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprob_mean = logprobs.mean(0)
    logprob_std = logprobs.var(0).sqrt()

    return logprob_mean, logprob_std

def train_eval_flow_2(flow, optimizer, train_loader, test_loader):
    """ train flow 2, learning p(I_0|E_inc) eval after each epoch"""
    # loop over epochs
    # train one step
    # evaluate
    # load best epoch in last step
    pass


###################################################################################################
#######################################   running the code   ######################################
###################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    LAYER_SIZE = {'2': 9 * 16, '3': 18 * 50}[args.which_ds]
    DEPTH = 45

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

    preprocessing_kwargs = {'with_noise': True, 'noise_level': 1e-4, 'apply_logit': True,
                            'do_normalization': True}

    if args.which_flow >= 1:
        print("Working on Flow 1")
        print("Working on Flow 1", file=open(args.results_file, 'a'))
        train_loader_1, test_loader_1 = get_calo_dataloader(
            os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)), 1, args.device,
            which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

        flow_1, optimizer_1 = build_flow(DEPTH, 1, args)

        if args.train:
            train_eval_flow_1(flow_1, optimizer_1, train_loader_1, test_loader_1, args)

        if args.evaluate:
            flow_1 = load_flow(flow_1, 1, args)
            logprob_mean, logprob_std = eval_flow_1(test_loader_1, flow_1, args)
            output = 'Evaluate (flow 1) -- ' +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std),
                  file=open(args.results_file, 'a'))

        if args.generate:
            flow_1 = load_flow(flow_1, 1, args)
            pass


    train_loader_2, test_loader_2 = get_calo_dataloader(
        os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)), 2, args.device,
        which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

    flow_2, optimizer_2 = build_flow(LAYER_SIZE, 1, args)

    train_eval_flow_2()
