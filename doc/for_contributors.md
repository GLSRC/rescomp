## For Contributors
### General Information
* As the package is very much in flux, reading the commits of others is very important! Otherwise you might write code for functions that don't exist anymore or whose syntax has completely changed.  
As a corrolary, this also means that writing legible, descriptive commit messages is paramount!

* All the old code, before we made everything installable as a package, is in the folder _legacy_. The goal should be to slowly add all the functions from the legacy code base to one, coherent python package.  


### Branche Usage Intentions 
* **master**: 
    Only for finished, documented features whose signature or functionality won't change much in the future  
    Changed only by merging with the development branch. Every change is accompanied by a new package version and updated documentation.  

* **develop**: 
    This is the development branch which is intendet for working features whose signature or functionality probably won't change much in the future, but are not ready yet to be deployed to "the public", possibly because they are incomplete or not yet documented.
    Also for features and changes to insignificant to make a new release number for.

* **developer branches**:
    These are your personal branches you can do what you want with.  
    They should branch off of and merge into the develop branch. It is advisable to keep your developer branch as synchronized as possible with the development branch (see below)


### Create your own developer branch
First, go into your local rescomp repository. Then, make sure that you are in the master branch, which you can check via:

    git status

Then, you want to make sure that your local repository is synchronized with the code on the GitLab servers, which you can enforce by using

    git pull

checkout (i.e. switch to) the developer branch
    
    git checkout develop
    
create your branch locally

    git branch Your-GitLab-Username-Here
    
and then push the branch to server

    git push --set-upstream origin Your-GitLab-Username-Here

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

Do **not** merge your personal _developer_ branches directly into master!  
Only the _development_ branch should be merged into the master branch directly.


### Testing
We now have tests which you can and should use to check if your changes break something unintended in the package.  
To run them, just enter the repository folder and use the command

    python -m unittest -v

Currently, the tests are rather sparse, so all contributions to the testing suite are highly encouraged


### Cloning the Repository via Access Token Authentication:

Under your profile's GitLab 'User Setting > Access Token' you can create an access token to allow individual machines access without your GitLab password. 
This removes the need to type your password every time you push/pull or, worse, the workaround to save your password somewhere on the machine itself.  

After your access token is created, you can clone the repository by using the command:

    git clone https://oauth2:YOUR-ACCESS-TOKEN@gitlab.de/rescom/reservoir-computing.git rescomp


### Code Style
In general, this repository follows the PEP8 Style Conventions [pep8 style guide](https://www.python.org/dev/peps/pep-0008/).  
Mostly this means all names should be lowercase with underscores, besides class names which should follow the CapWords convention. Also all names should be descriptive and represent what they are actually doing, wherever possible.


#### Docstrings:  
Please use the Google style docstring format to write your docstring. Only by using consistent docstring styles, can the html documentation be created automatically!  
For examples of Google style docstrings just look at the already existing docstring or visit:
* [google style example](https://www.sphinx-doc.org/en/1.6/ext/example_google.html)
* [why google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)


### For Internal Contributors
When you install the package on the Rechenknecht, instead of using the environment file "environment_rescomp.yml" specified above, use the much more detailed one for the Rechenknecht "environment_rescomp_full_rk.yml". The command to create the environment would then be
      
    conda env create --name rc_env --file environment_rescomp_full_rk.yml
      
Doing so should ensure absolute reproducability between all results calculated on the Rechenknecht.

