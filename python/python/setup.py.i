from setuptools import setup, find_packages

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

requirements = [
    'numpy',
    #'tqdm',
    #'requests',
    #'portalocker',
    #'opencv-python'
]

setup(
  name          = 'ncnn',
  version       = '${PACKAGE_VERSION}',
  url           = 'https://github.com/caishanli/pyncnn',
  packages      = find_packages(),
  package_dir   = {'': '.'},
  package_data  = {'ncnn': ['ncnn${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}']},
  install_requires = requirements
)
