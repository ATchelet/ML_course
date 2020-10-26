# ML_FOOLS team Project 1 "Higgs Boson Challenge"

## Files for final submission

In this folder you will find two principal files :
- run.py
> you can simply invoque it from command line, it will take a minute or so to complete, and it will create an output csv file in the same directory. There needs to be a test.csv and train.csv file in a data folder higer in the hierarchy like ../data/train.csv relative to the run.py file
- implementations.py
> Contains course functions that were asked to have available, you can run them each individually.

## Files in which we tested different approaches

### Jupyter notebooks

- th_ridge_trimmed19features.ipynb
> jupyter notebook where we tried ridge regression on only features with no errors throughout, of which there were 19. In the end split 8 sets was better.

- th_ridge_split8sets.ipynb
> jupyter notebook where we did some heavy hyper parameter searching for our 8 split datasets based on physics backgrounds (see https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf). The graphs, degrees and lambdas found helped us fine tune to a restricted lambdas set and restricted degrees.

- th_ridge_split8sets-restricted.ipynb
> jupyter notebook where we fine tuned the lambdas and degrees search to have reasonable compute times. This produced our best results, and is what we put into run.py in the end.

- th_ridge_split8sets-standardized.ipynb
> jupyter notebook where we tested standardization for our split sets, but didn't end up getting best results.

- th_ridge_split8sets-deg0-5.ipynb
> jupyter notebook where we speed tested degrees 0 to 5 to see if they had good results. In the end, higher degrees were necessary.

- th_ridge_split8sets-restricted-noretrain.ipynb
> file similar to th_ridge_split8sets-restricted but just testing an alternative last step, to not retrain our models from the hole dataset. This did not change the results, so we did not use it because it was not using as much data in the end.

- test_implementations
> Testing of crucial functions in the implementations.py file

- logistic_GD_with_cross_validation-project1.ipynb
> Looking at results given by logistic regression gradient descent method through a 6-fold cross validation

### Python files

- th_helpers.py
> some helper functions, for general data handling.

- th_ridge_regression.py
> small variant of ridge redgression in implementations, which just does store errors to improve time.

- proj1_helpers.py
> functions provided by course organisers to help us gain time on basic tasks.

- lab3_plots.py
> plotting functions from lab 3, useful for optimizing lambdas in ridge regression.


