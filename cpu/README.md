# Disorder order clAssifier by Rapid and User-friendly Machine, DARUMA
DARUMA (Disorder order clAssifier by Rapid and User-friendly Machine) is a predictor for intrinsically disordered regions(IDRs). 
DARUMA enables rapid prediction by using a simple convolutional neural network that uses physicochemical properties of amino acids as features.
Developed by Fukuchi Lab, Maebashi Institute of Technology.

## USAGE
    python3 ./predict.py [SEQUENCE FILE]
SEQUENCE FILE : Path to fasta formatted file(supports both multi-fasta and single-fasta).

## Installation
### Install package (Recommended)
    pip install daruma

### Install docker
    docker pull antepontem/daruma
