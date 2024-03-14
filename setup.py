import sys

from os import path
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='expressive-losses',
    description='Expressive Losses for Verified Robustness via Convex Combinations',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'torchvision', 'wandb', 'tqdm'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)
