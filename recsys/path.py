import os

filepath = os.path.realpath(__file__)
filedir = os.path.dirname(filepath)
AMAZON_DIR = os.path.dirname(filepath) + '/../dataset_preprocessing/amazon/files'