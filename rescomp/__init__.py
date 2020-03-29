# __init__.py

# from . import *
from . import simulations, utilities, measures, esn
from .esn import ESN, ESNWrapper
from .utilities import read_pickle
from ._version import __version__

if not utilities._compare_version_file_vs_env(segment_threshold="minor"):
    warn_string = \
        "The internal rescomp package version number '%s' does not match the " \
        "version number specified in the currently active python environment " \
        "'%s'.\nPlease update the package by entering the rescomp repository " \
        "folder and running: 'pip install --upgrade -e .'"

    import warnings
    warnings.filterwarnings("once", category=ImportWarning)
    warnings.warn(warn_string, ImportWarning, stacklevel=2)
    # meh = warnings.formatwarning(warn_string, ImportWarning, None, None, line="import rescomp")
    warnings.resetwarnings()
