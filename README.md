## Reservoir Computing

This repository contains the Pre-Alpha python 3 package **rescomp** implementing the machine learning technique Reservoir Computing(RC).
 
Development largely takes place at the DLR group _Komplexe Plasmen_ in Oberpfaffenhofen, but contributions from other DLR sites are encouraged.

For questions, feedback and ideas, [write us!][maintainer mail adresses]


## Licensing

As you might have noticed, this repository does not yet have a license file. This is on purpose, as we are currently discussing which licence would be appropriate for this project.  

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
      
  and then, should pip still not be located in the active environment by "which pip", install pip explicitly:
  
      conda install pip

* Nothing works, please help me!  

  If you can't install or use the package despite following the above instructions, [write us][maintainer mail adresses]. For a package as new as this one, such problems are to be expected and we will try to help as soon as possible.
  
### Uninstalling rescomp

To uninstall the rescomp package, simply activate the respective environment and then type:

    pip uninstall rescomp


## Usage

Please read the **FAQ below**.  

Otherwise, just look at the examples in _bin_ to get started. Not many features are implemented yet, hence there is not that much to explain.

### For Contributors

* As the package is very much in flux, reading the commits of others is very important! Otherwise you might write code for functions that don't exist anymore or whose syntax has completely changed.  
As a corrolary, this also means that writing legible, descriptive commit messages is paramount!

* All the old code, before we made everything installable as a package, is in the folder _legacy_. The goal should be to slowly add all the functions from the legacy code base to one, coherent python package.


#### For Internal Contributors
* When you install the package on the Rechenknecht, instead of using the environment file "environment_rescomp.yml" specified above, use the much more detailed one for the Rechenknecht "environment_rescomp_full_rk.yml". The command to create the environment would then be
      
      conda env create --name rc_env --file environment_rescomp_full_rk.yml
      
  Doing so should ensure absolute reproducability between all results calculated on the Rechenknecht.

* When importing a new packet, add it to the environment_rescomp.yml file, specifying the newest version that runs on the Rechenknecht, and to the setup.py under "install_requires", specifying the minimal version needed to run the code at all (usually the current major version)


## FAQ  
  
**Q:** Are the methods shown in the examples really all there is to the package?

**A:** While we have implemented many more features, ideas, etc., we chose not to add them to the package yet. We first want to re-write our current code base to be a foundation one can actually build a fully featured package on without having to re-write everything again in half a year due to a badly planned start.  

**Q:** My predictions don't agree with the real data at all! What is going on?
  
**A:** The performance of the reservoir depends strongly on the hyperparameters.  
Nobody really knows how _exactly_ the prediction quality depends on the parameters, but as a start you should use a reservoir that has about 100x as many nodes as the input has dimensions.  More for real or more "complicated" data.  
For all other parameters it is advisable to just play around by hand to see what parameter (ranges) might work or to use the hyperparameter optimization algorithm of your choice. As RC is fast to train, a simple grid search is often sufficient   
 
**Q:** For the exact same set of hyperparameters the prediction is sometimes amazingly accurate and at other times utterly wrong. Why?   

**A:** As the network is not trained at all, it can happen that the randomly generated network is not suited to learn the task at hand. How "good" and "bad" network topologies differ is an active research problem.  
Practically, this means that the prediction quality for a set of hyperparameters can not be determined by a single randomly generated network but instead must be calculated as e.g. the average of multiple randomly generated networks.  
Luckily, unsuccessful predictions are almost always very clearly distinguishable if the network topology is at fault, i.e. the prediction either works well or not at all. 

**Q:** You said above that the network should have about 100x as many nodes as the input has dimensions, maybe more. My input is >50 dimensional and with 5000-10000 nodes the training and prediction is annoyingly slow! I thought RC was supposed to be fast, what's going on?

**A:** The computational bottleneck of RC is a bunch of matrix multplications which, roughly, scale as O(n^3), where n is the number of nodes in the network. Therefore, just scaling up the network to accommodate larger and larger inputs doesn't work.  
Luckily there is a potential solution to this problem in the method of [local states][local states paper]. This is one of the features we already have code written for, but did not yet implemented in the package.  
If you need to use RC for higher dimensional inputs _now_, it is of course always helpful to reduce the input dimensionality as much as possible, for example via an autoencoder.

**Q:** Do I need to denoise my data?  

**A:** Yes, if you work with real data it should be denoised, removing as much noise as possible without loosing too much information. It's the same as for all other machine learning techniques, really.

**Q:** Your code is bad, your documentation is bad and you should feel bad. 
   
**A:** Well, this is quite rude. Also not a question.  
Nonetheless we do know that a lot of work needs to be done before the code can ascend to the status of "real python package". Regardless, we hope that you still get some use out of the package, even if it is just toying around with RC a bit while you wait for the code base to be developed further.


[maintainer mail adresses]: mailto:Jonas.Aumeier@dlr.de,Sebastian.Baur@dlr.de,Joschka.Herteux@dlr.de,Youssef.Mabrouk@dlr.de?cc=Christoph.Raeth@dlr.de
[rescomp gitlab link]: https://gitlab.dlr.de/rescom/reservoir-computing
[local states paper]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.024102