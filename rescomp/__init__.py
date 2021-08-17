# __init__.py

# from . import *
from . import simulations, utilities, measures, esn, locality_measures
from .esn import ESN, ESNWrapper, ESNGenLoc
from .utilities import read_pickle
from .simulations import simulate_trajectory
from ._version import __version__
import warnings

if not utilities._compare_version_file_vs_env(segment_threshold="minor"):
    int_version = utilities._get_internal_version()
    env_version = utilities._get_environment_version()

    warn_string = \
        "The internal rescomp package version '%s' does not match the " \
        "version specified in the currently active python environment " \
        "'%s'.\nPlease update the package by entering the rescomp repository " \
        "folder and running: 'pip install --upgrade -e .'"\
        %(int_version, env_version)

    warnings.filterwarnings("once", category=ImportWarning)
    warnings.warn(warn_string, ImportWarning, stacklevel=2)
    warnings.resetwarnings()
