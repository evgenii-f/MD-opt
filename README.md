## General info

This project is aimed to (re-)parameterize Force Fields (FFs) of classical Molecular Dynamic (MD) Simulations Bayesian Optimization (BO) coupled with Gaussian Processes (GP) (Active Learning approach).

## Technology
GROMACS
jupyter
numpy  
pandas  
scikit-learn  
scikit-optimizer  
GPy  
GPyopt  

## Setup 
To run the project, install packages from "requirement.txt" (pip installation : from terminal, cd to project directory and run "pip install -r requirements.txt"). After that, copy the project content to a given directory and run jupyter notebook "BOoptimization.npy"

GROMACS MD software must be installed independently. This framework iteratively runs MD code by executing "gmx ..." command in the terminal.

It is implied that all settings related to MD simulations and following post-processing routines are done preliminary by:
1. Preparing md_template directory, in which all simulation data and .mpd configuration files are kept. See GROMACS documentation https://manual.gromacs.org/current/user-guide/index.html.
2. Preparing a corresponding terminal command RUN_MD, see jJupyter Notebook BOoptimization.ipynb

## Usage
The optimization routine can be run via  BOoptimization.ipynb jupyter notebook. The following optimization results are saved at each iteration and can be monitored online:
1. Visualization of predictions, uncertanty and computed acquizition function over the search space. See projectDir/pics
2. Array predictions, uncertanty and computed acquizition function over the search space. See projectDir/history.npy
3. Optimization "path", or coordinates of points of the search space, which algorithm chose to check next. See projectDir/X.npy

## Test case
The project includes a test-case located in md_template folder. By running BOoptimization.ipynb with default settings, user can try BO optimization for 2 arbitrary FF parameters. 

