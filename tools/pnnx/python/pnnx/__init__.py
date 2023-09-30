import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from .pnnx import *

from .utils.export import export
from .utils.convert import convert

__version__ = pnnx.__version__
