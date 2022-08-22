# pylint: disable=invalid-name
""" Main script to run iterative flow for the CaloChallenge, datasets 2 and 3.

    by Claudius Krause, Matthew Buckley, Gopolang Mohlabeng, David Shih

"""


import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from nflows import transforms, distributions, flows

from data import get_calo_dataloader

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()

parser.add_argument()








normalization = {'2': 64172.594645065976, '3': 63606.50492698312} # or 6.5e4

preprocessing_kwargs = {'with_noise': False, 'noise_level': 1e-4, 'apply_logit': True,
                        'do_normalization': True, 'normalization': None}

