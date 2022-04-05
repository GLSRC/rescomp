## Installation

The following is a guide detailing how to install the rescomp package the way it "should" be done, i.e. as safely as possible.  
If you already know what you are doing, you can just install this package like any other locally installed package; no special dependencies are required.

These instructions are for unix systems, but do work with at most minor modifications on Windows too.

### Optional (but Strongly Encouraged) Prerequesites

* [git](https://git-scm.com/downloads): Used for downloading the repository and keeping it up to date. 
* [Anaconda 3](https://www.anaconda.com/distribution/): Using a virtual python environment is **highly** recommended to avoid damaging/unintentionally changing your base python.
  Of course you are free to use the environment manager of your choice, but in the following we will assume it to be Anaconda.

### Installation Instructions

Install git and Anaconda 3 (see above). Make sure to close and then reopen the terminal if this is the first time you installed Anaconda.

Open a terminal and enter the folder you wish to copy the GitLab repository to.  

Clone the GitLab repository. Here we copy it into the folder "rescomp"  

    git clone https://github.com/GLSRC/rescomp.git rescomp

Enter the cloned GitLab folder

    cd rescomp

Set your name and email for this repository, e.g.:
    
    git config user.name "Lastname, Firstname"
    git config user.email you@example.com

Create a new Anaconda environment _rc_env_ from the environment file included in the repository. (If you are on a DLR server/cluster, try using the full environment file _environment_rescomp_full_rk.yml_ first.)

    conda create --name rc_env --file requirements.txt

Activate your new environment

    conda activate rc_env

If you plan to contribute, or just want to change the code yourself, it is very convenient to install the package as an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) (the dot is important)
    
    pip install -e .

An editable install allows you to change the code in the repository and instantly work with the new behaviour without having the reinstall the package!  

Alternatively, to install everything like a normal, unchanging package, use (the dot is important here too)

    pip install .

After the installation, you can test that the package works correctly by installing pytest
    
    conda install pytest

and then running it:

    pytest

You should see a variety of tests being executed. If no big red errors are thrown, the installation was successful. 

After the installation, check out our [examples](examples.rst) to familiarize yourself with the package.  

### Common Installation Problems
####  The installation seems to have worked without error, but the package is not found when I try to import it. 

Make sure that pip is installed and active in the environment you want to install the package to:

    which pip

The output of the above should include the active environment name. For the environment _rc_env_ from above it should be something like

  > "/Users/<username>/anaconda3/envs/rc_env/bin/pip"

If it doesn't include the environment name, you installed the rescomp package in what ever environment the above specifies.   
To undo this, first uninstall the package,

    pip uninstall rescomp

then deactivate and activate your anaconda environment 

    conda deactivate
    conda activate rc_env

and then, should pip still not be located in the active environment by "which pip", install pip explicitly:

    conda install pip

#### Nothing works, please help me!  

If you can't install or use the package despite following the above instructions, [write us][maintainer mail adresses]. For a package as new as this one, such problems are to be expected and we will try to help as soon as possible.

### Updating

To keep the rescomp package up to date with the most current version in the repository, enter your local repository folder (the same folder we cloned during the installation) and run

    git pull
    
This updates your local copy of the package to be up to date with the one on the GitLab website.  
If you installed the package as an editable install during installation, you are done now.  
If you installed it as a regular package, you can finish the update by running

    pip install --upgrade .


### Uninstalling 

To uninstall the rescomp package, simply activate the respective environment and then type:

    pip uninstall rescomp

[maintainer mail adresses]: mailto:Sebastian.Baur@dlr.de?cc=Christoph.Raeth@dlr.de