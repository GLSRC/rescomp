## Reservoir Computing

This repository contains the Pre-Alpha python 3 package **rescomp** implementing the machine learning technique Reservoir Computing.
 
Development largely takes place at the DLR group _Komplexe Plasmen_ in Oberpfaffenhofen, but contributions from other DLR sites are encouraged.

For questions, feedback and ideas, [write us!][maintainer mail adresses]


## Licensing

As you might have noticed, this repository does not have a license file. This is on purpose, as we are currently discussing which licence would be appropriate for this project.  

As such we ask you not to share this code with _anyone_ that didn't explicitly get access to the [DLR Gitlab repository][rescomp gitlab link] via official channels until an appropriate license has been added.
 
 
## Installation

The following is a guide detailing how to install the rescomp package the way it "should" be done, i.e. as safely as possible.  
If you already know what you are doing, you can just install this package like any other locally installed package; no special dependencies are required.

These instructions are for unix systems, but should work with no or at most minor modifications on Windows too.

### Optional (but Strongly Encouraged) Prerequesites

* [git](https://git-scm.com/downloads): Used for downloading the repository and keeping it up to date. 
* [Anaconda 3](https://www.anaconda.com/distribution/): Using a virtual python environment is **highly** recommended to avoid irreparably damaging your system/working python as this is a Pre-Alpha distribution and hence not well tested.  
  Of course you are free to use the environment manager of your choice, but in the following we will assume it to be Anaconda.

### Installation Instructions

Install git and Anaconda 3 (see above). Make sure to close and then reopen the terminal if this is the first time you installed Anaconda.

Open a terminal and enter the folder you wish to copy the gitlab repository to.  

Clone the gitlab repository. Here we copy it into the folder "rescomp"  

    git clone https://gitlab.dlr.de/rescom/reservoir-computing.git rescomp

Enter the cloned gitlab folder

    cd rescomp

Set your (name and email for this repository (just use your full name and the DLR email address here)
    
    git config user.name "Lastname, Firstname"
    git config user.email you@example.com

Create a new Anaconda environment _rc_env_ from the environment file included in the repository. 

    conda env create --name rc_env --file environment_rescomp.yml

Activate your new environment

    conda activate rc_env

Create the package distribution files

    python setup.py sdist bdist_wheel

To install everything like a normal, unchanging package, use

    pip install .
 
If you plan to contribute, or just want to change the code yourself, it is much more conventient to install the package as an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) 
    
    pip install -e .

An editablte install allows you to change the code in the repository and instantly work with the new behaviour without having the reinstall the package! 


### Common Installation Problems

* The installation seems to have worked without error, but the package is not found when I try to import it. 

  Make sure that pip is installed and active in the environment you want to install the package to:
  
      which pip

  The output of the above should include the active environment name. For the environment _rc_env_ from above it should be something like

      > "/Users/<username>/anaconda3/envs/rc_env/bin/pip"

  If it doesn't include the environment name, you installed the rescomp package in what ever environment the above specifies.   
  To undo this, first uninstall the package,
  
      pip uninstall rescomp
      
  then deactivate and activate your anaconda environment 

      conda deactivate rc_env
      conda activate rc_env
      
  and then, should pip still not be located in the active environment by "_which pip_", install pip explicitly:
  
      conda install pip

* Nothing works, please help me!  

  If you can't install or use the package despite following the above instructions, [write us][maintainer mail adresses]. For a package as new as this one, such problems are to be expected and we will try to help as soon as possible.
  
### Uninstalling rescomp

To uninstall the rescomp package, simply activate the respective environment and then type:

    pip uninstall rescomp


## Usage

Please read the **FAQ** below.  

Otherwise, just look at the examples in _bin_ to get started. Not many features are implemented yet, hence there is not that much to explain.


## FAQ  
  
**Q:** My predictions don't agree with the real data at all! What is going on?
  
**A:** The performance of the reservoir depends strongly on the hyperparameters.  
Nobody really knows how _exactly_ the prediction quality depends on the parameters, but as a start you should use a reservoir that has about 10x-100x as many nodes as the input has dimensions.  More for real or more "complicated" data, less for synthetic inputs.    
For all other parameters it is advisable to just play around by hand to see what parameter (ranges) might work or to use the hyperparameter optimization algorithm of your choice. As RC is fast to train, a simple grid search is often sufficient   
 
**Q:** For the exact same set of hyperparameters the prediction is sometimes amazingly accurate and at other times utterly wrong. Why?   

**A:** As the network is not trained at all, it can happen that the randomly generated network is not suited to learn the task at hand. How "good" and "bad" network topologies differ is an active research problem.    
Practically, this means that the prediction quality for a set of hyperparameters can not be determined by a single randomly generated network but instead must be calculated using e.g. the success rate of at least 10 randomly generated networks, preferably more.    
Luckily, unsuccessful predictions are almost always very clearly distinguishable if the network topology is at fault, i.e. the prediction either works well or not at all. 

**Q:** Do I need to denoise my data?  

**A:** Yes, if you work with real data it should be denoised, removing as much noise as possible without loosing too much information. It's the same as for all other machine learning techniques, really.

**Q:** Your code is bad, your documentation is bad and you should feel bad. 
   
**A:** Well, this is quite rude. Also not a question.  
Nonetheless we do know that a lot of work needs to be done before the code can ascend to the status of "real python package". Regardless, we hope that you still get some use out of the package, even if it is just toying around with Reservoir Computing a bit while you wait for the code base to be developed further.


## For Contributors

* As the package is very much in flux, reading the commits of others is very important! Otherwise you might write code for functions that don't exist anymore or whose syntax has completely changed.  
As a corrolary, this also means that writing legible, descriptive commit messages is paramount!

* All the old code, before we made everything installable as a package, is in the folder _legacy_. The goal should be to slowly add all the functions from the legacy code base to one, coherent python package.


### For Internal Contributors
* When importing a new packet, add it to the environment_rescomp.yml file, specifying the newest version that runs on the Rechenknecht, as well as to the setup.py under "install_requires", specifying the minimal version needed to run the code at all (usually the last major update)



[maintainer mail adresses]: mailto:Jonas.Aumeier@dlr.de,Sebastian.Baur@dlr.de,Joschka.Herteux@dlr.de,Youssef.Mabrouk@dlr.de?cc=Christoph.Raeth@dlr.de
[rescomp gitlab link]: https://gitlab.dlr.de/rescom/reservoir-computing