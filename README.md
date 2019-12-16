# Reservoir Computing

This repository contains the python 3 package **rescomp** implementing the machine learning technique Reservoir Computing which is currently in the Pre-Alpha state.
 
Development largely takes place at the DLR group _Komplexe Plasmen_ in Oberpfaffenhofen, but contributions from other DLR sites are encouraged.

For questions, feedback and ideas, [write us!][all our mail adresses]

## Installation

The following is a guide detailing how to install the rescomp package the way it "should" be done, i.e. as safely as possible.  
If you already know what you are doing, you can just install this package like any other locally installed package though; no special dependencies are required.

#### Prerequesites

##### Absolutely Necessary
   * Python 3
   * A DLR GitLab account with access to this repository

##### Optional but encouraged

* [git](https://git-scm.com/downloads): Used for downloading the repository and keeping it up to date. 
* [Anaconda 3](https://www.anaconda.com/distribution/): Using a virtual python environment is **highly** recommended to avoid irreparably damaging your system/working python as this is a Pre-Alpha distribution and hence not well tested.  
  Of course you are free to use the environment manager of your choice, but in the following we will assume it to be Anaconda.

#### Installation Instructions
These instructions were written for unix systems/terminals, but should work essentially the same way on windows too.

Install git and Anaconda 3, see above.  

Open a terminal and enter the folder you wish to copy the gitlab repository to.  

Clone the gitlab repository. Here we copy it into the folder "rescomp"  

    git clone https://gitlab.dlr.de/rescom/reservoir-computing.git rescomp

Enter the cloned gitlab folder

    cd rescomp

Set your name and email for this repository
    
    git config user.name <Your Name>
    git config user.email <you@example.com>
    
Create a new Anaconda environment by cloning your base environment, your current working environment or just use the environment file we are working with. Here, we create the new environment _rc_env_ from the environment file included in the repository

    conda env create --name rc_env --file environment_rescomp.yml

Create the package distribution files

    python3 setup.py sdist bdist_wheel

Make sure that the is installed in the environment you want to install the package to:
    
    which pip

The output of the above should be something like 
    
    "/Users/<username>/anaconda3/envs/rc_env/bin/pip"
 
To install everything like a normal, unchanging package, use

    pip install .
    
If you plan to contribute, it is probably better to install the package asan [editable install](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode) 
    
    pip install -e .

An editablte install allows you to change the code in the repository and instantly work with the new behaviour without having the reinstall the package! 


### Un-Installation

To uninstall the rescomp package, simply activate the respective environment and type:

    pip uninstall rescomp

## Usage

Please read the **FAQ** below.  
Otherwise, just look at the examples in bin to get started. Not many features are implemented yet, hence there is not much to explain.

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
  * Well, this is quite rude. Also not a question.  
  Nonetheless we do know that a lot of work needs to be done before the code can ascend to the status of "real python package". Regardless, we hope that you still get some use out of the package, even if it is just toying around with Reservoir Computing a bit while you wait for the code base to be developed further.


## For Contributors

* As the package is very much in flux, reading the commits of others is of highest priority! Otherwise you might write code for functions that don't exist anymore or whose syntax has completely changed.  
As a corrolary, this also means that writing legible, high information content commit messages is paramount too!

* When importing a new packet, add it to the environment_rescomp.yml file, specifying the newest version that runs on the Rechenknecht, as well as to the setup.py under "install_requires", specifying the minimal version needed to run the code at all (usually the last major update)

* All the old code, before we made everything installable as a package, is in the folder "legacy". The goal should be to slowly add all the functions from the legacy code base to one, coherent python package.


[all our mail adresses]: mailto:sebastian.baur@dlr.de
[rescomp gitlab link]: https://gitlab.dlr.de/rescom/reservoir-computing