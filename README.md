## Reservoir Computing

This repository contains the Alpha python 3 package **rescomp** implementing the machine learning technique Reservoir Computing (RC).
 
Development largely takes place at the DLR group _Komplexe Plasmen_ in Oberpfaffenhofen, but contributions from other DLR sites are encouraged.

For questions, feedback and ideas, [write us!][maintainer mail adresses]


### Licensing

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

Create a new Anaconda environment _rc_env_ from the environment file included in the repository. (If you are on a DLR server/cluster, try using the full environment file _environment_rescomp_full_rk.yml_ first.)

    conda env create --name rc_env --file environment_rescomp.yml

Activate your new environment

    conda activate rc_env

Create the package distribution files

    python setup.py sdist bdist_wheel

If you plan to contribute, or just want to change the code yourself, it is very convenient to install the package as an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) (the dot is important)
    
    pip install -e .

An editable install allows you to change the code in the repository and instantly work with the new behaviour without having the reinstall the package!  

Alternatively, to install everything like a normal, unchanging package, use (the dot is important here too)

    pip install .


### Uninstalling rescomp

To uninstall the rescomp package, simply activate the respective environment and then type:

    pip uninstall rescomp


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


## Usage
### Getting Started
To get started, look at the examples in _bin_. The intended order is:  
1. minimal_example.py  
2. higher_dim_example.py  
3. measures_example.py  
4. utilities_example.py  

To learn more about the physics behind RC and what to expect/not expect from this method/package, please read the **FAQ below**.  


### Documentation
The code itself is documented in our html documentation. You can read it by downloaded/cloning this repository and opening the file *index.html* in the folder *doc_html* in your favorite web browser.  

In the future, this documentation will be hosted on a GitLab Pages website as well.


### FAQ  
#### Q: Where do I find good literature about reservoir computing ? Do you have some paper or book to recommend?  

**A:** For a comparison between different RNN methods and RC, as well as a demonstration of RC's predictive power and speed:  
*Chattopadhyay, A.; Hassanzadeh, P.; Palem, K.; Subramanian, D. Data-Driven Prediction of a Multi-Scale Lorenz 96 Chaotic System Using a Hierarchy of Deep Learning Methods: Reservoir Computing, ANN, and RNN-LSTM. 2019, 1–21.*

For a quick introduction:  
*Haluszczynski, A.; Räth, C. Good and Bad Predictions: Assessing and Improving the Replication of Chaotic Attractors by Means of Reservoir Computing. Chaos 2019, 29.*  
It explains the basic math quite well and applies RC to the canonical Lorenz system. Furthermore, it discusses possibly the biggest distinguishing feature of RC, the static randomness of it's network and the resulting consequences, which is very important to understand when working with RC.

For some "classical" RC literature, the paper(s) by Jaeger et. al.:  
*Lukoševičius, M.; Jaeger, H. Reservoir Computing Approaches to Recurrent Neural Network Training. Comput. Sci. Rev. 2009, 3, 127–149.*  
It discusses the basics and the many possible variations of RC in addition to its development history. This paper is of course not quite up to date anymore, as it's 10 years old by now, but the qualitative results and ideas still hold true today.

#### Q: My predictions don't agree with the real data at all! What is going on?  
  
**A:** The performance of the reservoir depends strongly on the hyperparameters.  
Nobody really knows how _exactly_ the prediction quality depends on the parameters, but as a start you should use a reservoir that has about 100x as many nodes as the input has dimensions.  More for real or more "complicated" data.  
For all other parameters it is advisable to just play around by hand to see what parameter (ranges) might work or to use the hyperparameter optimization algorithm of your choice. As RC is fast to train, a simple grid search is often sufficient   
 
#### Q: For the exact same set of hyperparameters the prediction is sometimes amazingly accurate and at other times utterly wrong. Why?  

**A:** As the network is not trained at all, it can happen that the randomly generated network is not suited to learn the task at hand. How "good" and "bad" network topologies differ is an active research problem.  
Practically, this means that the prediction quality for a set of hyperparameters can not be determined by a single randomly generated network but instead must be calculated as e.g. the average of multiple randomly generated networks.  
Luckily, unsuccessful predictions are almost always very clearly distinguishable if the network topology is at fault, i.e. the prediction either works well or not at all. 

#### Q: You said above that the network should have about 100x as many nodes as the input has dimensions, maybe more. My input is >50 dimensional and with 5000-10000 nodes the training and prediction is annoyingly slow! I thought RC was supposed to be fast, what's going on?  

**A:** The computational bottleneck of RC is a bunch of matrix multplications which, roughly, scale as O(n^3), where n is the number of nodes in the network. Therefore, just scaling up the network to accommodate larger and larger inputs doesn't work.  
Luckily there is a potential solution to this problem in the method of [local states][local states paper]. This is one of the features we already have code written for, but did not yet implemented in the package.  
If you need to use RC for higher dimensional inputs _now_, it is of course always helpful to reduce the input dimensionality as much as possible, for example via an autoencoder.

#### Q: In your code, I see you discard some samples at the beginning of training, why is it necessary?  

**A:** this is done for two reasons:

1\. To get rid of the transient dynamics in the reservoir, i.e. to "synchronize" the reservoir state r(t) with the trajectory, before the training starts. 
If one were to omit this step, the initial reservoir state would not correspond to the initial position on the trajectory, resulting in erroneous training results. For more details, i'd recommend reading up on the "echo state property" of RC and Echo State Networks in general.

2\. To discard any transient behavior in the training data itself. 
When trying to predict a chaotic attractor, the training data should only contain points on that attractor, otherwise the neural network learns the (irrelevant) dynamics of the system before the attractor is reached, which will decrease it's predictive quality on the attractor itself. 

Discarding the transient behavior in the data (2.) basically just amounts to data preprocessing and is not necessary for RC per se (one could always just use input data that does already start on the attractor) but as the synchronization between reservoir and trajectory (1.) is necessary, we use the "discard_steps" variable to accomplish both.

#### Q: Do I need to denoise my data?  

**A:** Yes, if you work with real data it should be denoised, removing as much noise as possible without loosing too much information. It's the same as for all other machine learning techniques, really.


## For Contributors
### General Information
* As the package is very much in flux, reading the commits of others is very important! Otherwise you might write code for functions that don't exist anymore or whose syntax has completely changed.  
As a corrolary, this also means that writing legible, descriptive commit messages is paramount!

* All the old code, before we made everything installable as a package, is in the folder _legacy_. The goal should be to slowly add all the functions from the legacy code base to one, coherent python package.  


### Branche Usage Intentions:  
* **master**: 
    Only for finished, documented features whose signature or functionality won't change much in the future  
    Changed only by merging with the development branch. Every change is accompanied by a new package version and updated documentation.  

* **develop**: 
    This is the development branch which is intendet for working features whose signature or functionality probably won't change much in the future, but are not ready yet to be deployed to "the public", possibly because they are incomplete or not yet documented.
    Also for features and changes to insignificant to make a new release number for.

* **developer branches**:
    These are your personal branches you can do what you want with.  
    They should branch off of and merge into the develop branch. It is advisable to keep your developer branch as synchronized as possible with the development branch (see below)


### Create your own developer branch:
First, go into your local rescomp repository. Then, make sure that you are in the master branch, which you can check via:

    git status

Then, you want to make sure that your local repository is synchronized with the code on the Gitlab servers, which you can enforce by using

    git pull

checkout (i.e. switch to) the developer branch
    
    git checkout develop
    
create your branch locally

    git branch Your-Gitlab-Username-Here
    
and then push the branch to server

    git push --set-upstream origin Your-Gitlab-Username-Here

after which you are done.
You now have a personal copy of the develop branch, which you can modify to you hearts content without getting in the way of other people!  
Now you just need to make sure that when you are working on/with the rescomp package, that you are in fact working in your own branch. This is checkable manually via 
    
    git status 
    
but you can also adjust your terminal to always display the current active branch, which is much safer and more convenient.


### Adjusting the terminal to always show the current branch

The precise way to do this depends on your operating system and shell.  
For bash on linux the following tutorial works quite well:
https://coderwall.com/p/fasnya/add-git-branch-name-to-bash-prompt


### Synchronize and merge your developer branch

To synchronize your developer branch with the current state of the development branch do:

    git checkout develop
    git pull
    git checkout Your-Developer-Branch-Name
    git merge develop

To merge your new, commit code from your personal developer branch into the development branch, first synchronize them as above:

    git checkout develop
    git pull
    git checkout Your-Developer-Branch-Name
    git merge develop

and then do:

    git checkout develop
    git merge Your-Developer-Branch-Name
    git push

Do **not** merge developer branches into master!  
Only the development branch should be merged into master, and please only do so if you really know what you are doing.


### Testing:  
We now have tests which you can and should use to check if your changes break something unintended in the package.  
To run them, just enter the repository folder and use the command

    python -m unittest -v

Currently, the tests are rather sparse, so all contributions to the testing suite are highly encouraged


### Docstrings:  
Please use the Google style docstring format to write your docstring. Only by using consistent docstring styles, can the html documentation be created automatically!  
For examples of Google style docstrings just look at the already existing docstring or visit:
* [google style example](https://www.sphinx-doc.org/en/1.6/ext/example_google.html)
* [why google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)


### For Internal Contributors
When you install the package on the Rechenknecht, instead of using the environment file "environment_rescomp.yml" specified above, use the much more detailed one for the Rechenknecht "environment_rescomp_full_rk.yml". The command to create the environment would then be
      
    conda env create --name rc_env --file environment_rescomp_full_rk.yml
      
Doing so should ensure absolute reproducability between all results calculated on the Rechenknecht.


[maintainer mail adresses]: mailto:Jonas.Aumeier@dlr.de,Sebastian.Baur@dlr.de,Joschka.Herteux@dlr.de,Youssef.Mabrouk@dlr.de?cc=Christoph.Raeth@dlr.de
[rescomp gitlab link]: https://gitlab.dlr.de/rescom/reservoir-computing
[local states paper]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.024102
