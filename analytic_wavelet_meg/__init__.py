from . import aggregation
from . import element_analysis_meg
from . import kendall_tau_module

from .aggregation import *
from .element_analysis_meg import *
from .kendall_tau_module import *

__all__ = ['aggregation', 'element_analysis_meg', 'kendall_tau_module']

__all__.extend(aggregation.__all__)
__all__.extend(element_analysis_meg.__all__)
__all__.extend(kendall_tau_module.__all__)
