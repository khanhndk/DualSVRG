from __future__ import absolute_import

from . import configs
from .model import Model
from .optimizer import Optimizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
try:
    from .tensorflow_model import TensorFlowModel
except ImportError:
    TensorFlowModel = None

__version__ = '0.1.0'
