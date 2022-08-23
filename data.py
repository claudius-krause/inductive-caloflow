# pylint: disable=invalid-name
""" Dataloader for calorimeter data of the CaloChallenge, datasets 2 and 3.

    by Claudius Krause, Matthew Buckley, Gopolang Mohlabeng, David Shih

"""

import os

import numpy as np
import h5py
import torch

from torch.utils.data import Dataset, DataLoader

def add_noise(input_array, noise_level=1e-4):
    noise = np.random.rand(*input_array.shape)*noise_level
    return (input_array+noise)/(1.+noise_level)

ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return np.log(x / (1.0 - x))

def sigmoid(x):
    """ returns sigmoid of input """
    return np.exp(x) / (np.exp(x) + 1.)

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clip(clamp_low, clamp_high)

class CaloDataLayerEnergy(Dataset):
    """ Dataloader for E_i of each layer (flow-1)"""
    def __init__(self, path_to_file, which_ds='2',
                 beginning_idx=0, data_length=100000,
                 **preprocessing_kwargs):
        """
        Args:
            path_to_file (string): path to .hdf5 file
            which_ds ('2' or '3'): which dataset (kind of redundant with path_to_file name)
            beginning_idx (int): at which index to start taking the data from
            data_length (int): how many events to take
            preprocessing_kwargs (dict): dictionary containing parameters for preprocessing
                                         (with_noise, noise_level, apply_logit, do_normalization,
                                          normalization)
        """

        # dataset specific
        self.layer_size = {'2': 9 * 16, '3': 18 * 50}[which_ds]
        self.depth = 45

        self.full_file = h5py.File(path_to_file, 'r')

        self.noise_level = preprocessing_kwargs.get('noise_level', 1e-4) # in MeV
        #self.apply_logit = preprocessing_kwargs.get('apply_logit', True)
        self.with_noise = preprocessing_kwargs.get('with_noise', True)
        #self.normalization = preprocessing_kwargs.get('normalization', None)

        showers = self.full_file['showers'][beginning_idx:beginning_idx+data_length]
        incident_energies = self.full_file['incident_energies']\
            [beginning_idx:beginning_idx+data_length]
        self.full_file.close()

        self.E_dep = showers.reshape(-1, self.depth, self.layer_size).sum(axis=-1)

        self.E_inc = incident_energies

    def __len__(self):
        # assuming file was written correctly
        return len(self.E_dep)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        energy_dep = self.E_dep[idx]
        e_inc = self.E_inc[idx]
        if self.with_noise:
            energy_dep = add_noise(energy_dep, noise_level=self.noise_level)

        sample = {'energy_dep': energy_dep, 'energy_inc':e_inc}

        return sample


class CaloDataShowerShape(Dataset):
    """ Dataloader for every Calorimeter Layer (flow-2 for layer0 and flow-3 for iterative step)"""
    def __init__(self, path_to_file, which_ds='2', which_layer=-1,
                 beginning_idx=0, data_length=100000,
                 **preprocessing_kwargs):
        """
        Args:
            path_to_file (string): path to .hdf5 file
            which_ds ('2' or '3'): which dataset (kind of redundant with path_to_file name)
            which_layer (int): which layers to include in data. -1 stands for all >0, non-negative
                               integers return data of that layer alone. None returns all.
            beginning_idx (int): at which index to start taking the data from
            data_length (int): how many events to take
            preprocessing_kwargs (dict): dictionary containing parameters for preprocessing
                                         (with_noise, noise_level, apply_logit, do_normalization,
                                          normalization)
        """

        # dataset specific
        self.layer_size = {'2': 9 * 16, '3': 18 * 50}[which_ds]
        self.depth = 45

        self.full_file = h5py.File(path_to_file, 'r')

        # effectively a flow-2 vs flow-3 switch
        self.which_layer = which_layer

        self.noise_level = preprocessing_kwargs.get('noise_level', 1e-4) # in MeV
        self.apply_logit = preprocessing_kwargs.get('apply_logit', True)
        self.with_noise = preprocessing_kwargs.get('with_noise', True)
        self.do_normalization = preprocessing_kwargs.get('do_normalization', True)
        #self.normalization = preprocessing_kwargs.get('normalization', 1.)


        showers = self.full_file['showers'][beginning_idx:beginning_idx+data_length]
        incident_energies = self.full_file['incident_energies']\
            [beginning_idx:beginning_idx+data_length]
        self.full_file.close()

        self.E_dep = showers.reshape(-1, self.depth, self.layer_size).sum(axis=-1)

        self.E_inc = incident_energies.repeat(self.depth, 0)

        # sum the deposited energy of each layer, then offset by one so each layer dataitem has
        # the deposited energy in the previous layer (_p). 0th layer gets zero energy deposited.
        self.E_dep_p = np.zeros_like(self.E_dep)
        self.E_dep_p[:, 1:] = self.E_dep[:, :-1]

        self.E_dep_p = self.E_dep_p.reshape(-1, 1)
        self.E_dep = self.E_dep.reshape(-1, 1)

        #get the layers, and normalize
        self.layer = showers.reshape(-1, self.depth, self.layer_size)


        #get the previous layers, and normalize
        #layer 0 gets zeros for the previous layer
        self.layer_p = np.zeros_like(self.layer)
        self.layer_p[:, 1:, :] = self.layer[:, :-1, :]

        self.layer = self.layer.reshape(-1, self.layer_size)
        self.layer_p = self.layer_p.reshape(-1, self.layer_size)

        self.layer_number = np.tile(range(self.depth), int(len(self.E_inc)/self.depth))

        if self.which_layer == -1:
            mask = (self.layer_number > 0)
        elif self.which_layer > -1:
            mask = (self.layer_number == self.which_layer)
        if which_layer is not None:
            self.layer_number = self.layer_number[mask]
            self.E_inc = self.E_inc[mask]
            self.E_dep = self.E_dep[mask]
            self.E_dep_p = self.E_dep_p[mask]
            self.layer = self.layer[mask]
            self.layer_p = self.layer_p[mask]

    def __len__(self):
        """ length of dataset should be length of E_inc """
        return len(self.E_inc)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        layer = self.layer[idx]
        layer_p = self.layer_p[idx]
        energy = self.E_inc[idx]
        energy_dep = self.E_dep[idx]
        energy_dep_p = self.E_dep_p[idx]
        layer_number = self.layer_number[idx]
        if self.with_noise:
            energy_dep = add_noise(energy_dep, noise_level=self.noise_level)
            layer = add_noise(layer, noise_level=self.noise_level) #/self.normalization)

            if self.which_layer != 0:
                energy_dep_p = add_noise(energy_dep_p, noise_level=self.noise_level)
                layer_p = add_noise(layer_p, noise_level=self.noise_level) #/self.normalization)

        if self.do_normalization:
            layer = layer/(energy_dep+self.noise_level)
            layer_p = layer_p/(energy_dep_p+self.noise_level)

            #energy_dep = energy_dep/self.normalization
            #energy_depn = energy_depn/self.normalization

        if self.apply_logit:
            layer = logit_trafo(layer)
            #energy_dep = logit_trafo(energy_dep)
            if self.which_layer != 0:
                layer_p = logit_trafo(layer_p)
                #energy_dep_p = logit_trafo(energy_depn)

        sample = {'layer': layer, 'layer_p': layer_p, 'energy': energy,
                  'energy_dep': energy_dep, 'energy_dep_p': energy_dep_p,
                  'layer_number': layer_number}

        return sample

def get_calo_dataloader(path_to_file, which_flow, device, which_ds='2', batch_size=32,
                        **preprocessing_kwargs):
    """ returns train/test dataloader for training each of the flows """
    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

    data_length = 100000
    train_length = int(0.7*data_length)
    test_length = int(0.3*data_length)

    if which_flow == 1:
        train_dataset = CaloDataLayerEnergy(path_to_file, which_ds=which_ds,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = CaloDataLayerEnergy(path_to_file, which_ds=which_ds,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)
    elif which_flow == 2:
        train_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds, which_layer=0,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds, which_layer=0,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)
    elif which_flow == 3:
        train_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds, which_layer=-1,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds, which_layer=-1,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, **kwargs)
    return train_dataloader, test_dataloader
