from setuptools import setup, find_packages
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, 'recsys'))
from version import __version__

print('version')
print(__version__)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='recsys',
      version=__version__,
      description='Recommender systems for evolving graphs',
      url='https://github.com/weihua916/dyn-recsys',
      author='Weihua Hu',
      author_email='weihua916@gmail.com',
      keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires = [
        'torch>=1.7.0',
        'numpy>=1.16.0',
        'tqdm>=4.29.0',
        'scikit-learn>=0.20.0',
        'pandas>=0.24.0',
        'six>=1.12.0',
        'urllib3>=1.24.0',
        'outdated>=0.2.0'
      ],
      license='MIT',
      packages=find_packages(exclude=['dataset_preprocessing']),
      package_data={'recsys': []},
      include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
    ],
)