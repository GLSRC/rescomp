"""
"""

# from distutils.core import setup
import setuptools
from setuptools import setup
from rescomp._version import __version__

setup(
    name='rescomp',
    version=__version__,
    description='Reservoir Computing package developed at the DLR',
    license_files=('LICENSE.txt',),
    author='Jonas Aumeier, Sebastian Baur, Joschka Herteux, Youssef Mabrouk',
    author_email='Jonas.Aumeier@dlr.de, Sebastian.Baur@dlr.de, Joschka.Herteux@dlr.de, Youssef.Mabrouk@dlr.de',
    maintainer='Sebastian.Baur',
    maintainer_email='Sebastian.Baur@dlr.de',
    url='https://github.com/GLSRC/rescomp',
    download_url='https://github.com/GLSRC/rescomp.git',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Operating System :: OS Independent"
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    # In the interest of ease of use, the highest package versions supported and tested are not specified here.
    # If an upper version limit is specified, that means there is a known bug with that version
    install_requires=[
        'numpy>=1.14.5',
        'networkx>=2.0.0',
        'pandas>=1.0.0',
        'scipy>=1.4.0',
        'scikit-learn>=0.23.0',
    ],
    provides=['rescomp']
)
