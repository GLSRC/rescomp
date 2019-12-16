# Reservoir Computing

This [repository][rescomp gitlab link] contains the python 3 package **rescomp** implementing the machine learning technique Reservoir Computing which is currently in the Pre-Alpha state.
 
Development largely takes place at the DLR group _Komplexe Plasmen_ in Oberpfaffenhofen, but contributions from other DLR sites are encouraged.



## Installation

The following is a guide detailing how to install the rescomp package the way it "should" be done, i.e. as safely as possible.  
If you already know what you are doing, you can just install this package like any other locally installed package though; no special dependencies are required.

### Prerequesites

Besides python 3, the following programs are encouraged: 

* [git](https://git-scm.com/downloads): Used for downloading the repository and keeping it up to date. 
* [Anaconda 3](https://www.anaconda.com/distribution/): Using a virtual python environment is **highly** recommended to avoid irreparably damaging your system/working python as this is a Pre-Alpha distribution and hence not well tested.  
  Of course you are free to use the environment manager of your choice, but in the following we will assume it to be Anaconda.

### Installation Instructions
These instructions were written for unix systems/terminals, but should work essentially the same way on windows too.

0. Install Git and Anaconda 3
1. Create a new Anaconda environment by cloning your base, or current working environment. For the base environment use:
> 
install anaconda

make new conda environment, cloning the base environment or your environment of choice

git clone

cd into cloned gitlab folder

python3 setup.py sdist bdist_wheel

which pip 
    make sure the pip is installed in the environment you wnat to install the package to, i.e. it should be something like:
    /Users/[username]/anaconda3/envs/[environment name]/bin/pip
    
pip install dist/rescomp-[version number].tar.gz
or pip install -e .
    "editable mode" https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode

### Un-Installation

pip uninstall rescomp

## Usage

**PLEASE** read the **FAQ** below.  
Otherwise, just look at the examples to get started.

## FAQ

* My predictions don't agree with the real data at all! What is going on?  
  * The performance of the reservoir depends strongly on the hyperparameters.  
  Nobody really knows _how exactly_ the prediction quality depends on the parameters, but as a start you should use a reservoir that has about 10x-100x as many nodes as the input has dimensions.  More for real or more "complicated" data, less for synthetic inputs.  
  For all other parameters it is advisable to just play around by hand to see what parameter (ranges) might work or to use the hyperparameter optimization algorithm of your choice. As RC is fast to train, a simple grid search is often sufficient.
  
* For the exact same set of hyperparameters the prediction is sometimes amazingly accurate and at other times utterly wrong. Why?
  * As the network is not trained at all, it can happen that the randomly generated network is not suited to learn the task at hand. How "good" and "bad" network topologies differ is an active research problem.  
  Practically, this means that the prediction quality for a set of hyperparameters can not be determined by a single randomly generated network but instead must be calculated using e.g. the success rate of at least 10 randomly generated networks, preferably more.  
  Luckily, unsuccessful predictions are almost always very clearly distinguishable if the network topology is at fault, i.e. the prediction either works well or not at all.
 
* Do I need to denoise my data?
  * Yes, if you work with real data it should be denoised as strongly as possible without loosing too much information. It's the same as for all other ML techniques, really.
  
* Your code is bad, your documentation is bad and you should feel bad.
  * Well, this is quite rude and also not a question..  
  Nonetheless we do know that a lot of work needs to be done before it can ascend to the status of "real python package". Regardless, we hope that you still get some use out of the package, even if it is just toying around with RC for now while you wait for the code base to be developed further


[rescomp gitlab link]: https://gitlab.dlr.de/rescom/reservoir-computing