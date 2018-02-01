# Data handler module

import urllib.request
import zipfile
import os

import pandas as pd
import numpy as np

# download data file
def download_data():
    # list of required files
    file_list = ['Train/X_train.txt','Train/y_train.txt',\
        'Test/X_test.txt','Test/y_test.txt','activity_labels.txt']

    # check if files are already resent
    all_files_present = True
    for fpath in file_list:
        all_files_present &= os.path.isfile(os.path.join('data', fpath)) 

    if not all_files_present:
        
        # create folders
        if not os.path.exists('data/Train'):
            os.makedirs('data/Train')
        if not os.path.exists('data/Test'):
            os.makedirs('data/Test')

        # download zip
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
            +'00341/HAPT%20Data%20Set.zip'
        print('Downloading data (79.6 MB)...')
        filehandle, _ = urllib.request.urlretrieve(url)
        
        # option for manually downloaded archive
        # filehandle = open('data/HAPT Data Set.zip','rb')

        # extract relevant files
        for fpath in file_list:
            extract_file(filehandle,fpath)
        print('Done.')
    else:
        print('Data checked.')

# extract a specific file from the zip
def extract_file(filehandle,dataname):
	
    with zipfile.ZipFile(filehandle) as z:
        with open(os.path.join('data', dataname), 'wb') as f:
            f.write(z.read(dataname))

# read data
def read_data():

    # download data if necessary
    download_data()

    # read data
    X_train = pd.read_csv('data/Train/X_train.txt',delim_whitespace=True,header=None)
    y_train = pd.read_csv('data/Train/y_train.txt',delim_whitespace=True,header=None)
    X_test = pd.read_csv('data/Test/X_test.txt',delim_whitespace=True,header=None)
    y_test = pd.read_csv('data/Test/y_test.txt',delim_whitespace=True,header=None)

    return X_train, X_test, y_train, y_test

# read activity labels
def read_activities():
    classes = pd.read_csv('data/activity_labels.txt',
        delim_whitespace=True, # use blank spaces as separator
        header=None,
        names=('ID','activity')) # use instead these names as header

    return classes
