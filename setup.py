"""
"""

# from distutils.core import setup
import setuptools
from setuptools import setup
from rescomp._version import __version__

setup(
    name='rescomp',
    # version='0.0.2',
    version=__version__,
    description='Reservoir Computing package developed at the DLR',
    # long_description=long_description,  # TODO
    # long_description_content_type="text/markdown",
    # keywords='',  # TODO
    author='Jonas.Aumeier, Sebastian.Baur, Joschka.Herteux, Youssef.Mabrouk',
    author_email='Jonas.Aumeier@dlr.de, Sebastian.Baur@dlr.de, Joschka.Herteux@dlr.de, Youssef.Mabrouk@dlr.de',
    maintainer='Jonas.Aumeier, Sebastian.Baur, Joschka.Herteux, Youssef.Mabrouk',
    maintainer_email='Jonas.Aumeier@dlr.de, Sebastian.Baur@dlr.de, Joschka.Herteux@dlr.de, Youssef.Mabrouk@dlr.de',
    url='https://gitlab.dlr.de/rescom/reservoir-computing',
    download_url='https://gitlab.dlr.de/rescom/reservoir-computing.git',
    # package_dir = {'': 'src'},
    # packages=['rescomp'],
    packages=setuptools.find_packages(),
    # scripts=[],  # TODO
    # license='',  # TODO
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: GNU General Public License (GPL)', # TODO
        'Natural Language :: English',
        # 'Operating System :: POSIX',
        # 'Operating System :: POSIX :: BSD :: FreeBSD',
        # 'Operating System :: POSIX :: BSD :: OpenBSD',
        # 'Operating System :: POSIX :: Linux',
        # 'Operating System :: Unix',
        # 'Operating System :: MacOS',
        "Operating System :: OS Independent"
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        # 'Topic :: Software Development :: Libraries :: Python Modules' # TODO
    ],
    install_requires=[
        'numpy>=1',
        'scipy>=1',
        'networkx>=2'
    ],
    provides=['rescomp']
)
