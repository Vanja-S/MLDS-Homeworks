# HW1

## Instructions

For this homework, you will implement classification trees and random forests that support numeric input variables and a binary target variable.

[PART 1]

You will implement classification trees and random forests as classes (Tree, RandomForest) that provide a method build, which returns the model as an object, whose predict method returns the predicted target class of given input samples (see attached code for usage examples):

Tree - a flexible (greedy) classification tree with the following attributes: (1) rand, a random generator, for reproducibility, of type random.Random; (2) get_candidate_columns, a function that returns a list of column indices considered for a split (needed for random forests); and (3) min_samples, the minimum number of samples for which a node is still split further. Use the Gini impurity to select the best splits.

RandomForest, with attributes: (1) rand, a random generator; (2) n: number of
bootstrap samples (trees). The RandomForest should use an instance of Tree internally. Build full trees (min_samples=2). For each split, consider a random subset of variables of size equal to the square root of the number of input variables.

Test your implementation with unit tests that focus on the critical or hard parts and edge cases. Combine all tests in a class named MyTests.

Apply the developed methods to the TKI resistance FTIR spectral dataset. Always use the -train data as the training set and the -test data as the testing set. Do the following:

In function hw_tree_full, build a tree with min_samples=2. Return misclassification rates and standard errors when using training and testing data as test sets.

In function hw_randomforest, use random forests with n=100 trees with min_samples=2. Return misclassification rates and standard errors when using training and testing data as test sets.

As a rough guideline, building the full tree on this dataset should take less than 10 seconds - more indicates inefficiencies in the implementation.

This assignment requires that you compute standard errors to quantify the uncertainty of the misclassification rates. Here, we only require an estimate of the uncertainty stemming from a particular test set measurement. Therefore, there is no need to rebuild models when computing standard errors for this assignment.

In the report:

Explain how you quantify the uncertainty of your estimates.

Show misclassification rates (and their uncertainties) from hw_tree_full.

Show misclassification rates (and their uncertainties) from hw_randomforest.

Plot misclassification rates versus the number of trees n.

[PART 2, grades 7-8]

Implement permutation-based variable importance. Refer to the "Variable Importance" section from "The Elements of Statistical Learning"; implement it as method importance() of the random forest model.

Computing random forest variable importance for all variables should not be a slow operation. For me, it is much faster than building the random forest.

In the report, plot variable importance for the given dataset for an RF with n=100 trees. Note that variables have a particular ordering for this kind of data, so keep the variable order when plotting. For comparison, also show variables from the roots of 100 non-random trees on sensibly randomized data on the same plot.

[PART 3, grades 9-10]

Devise, implement, and evaluate classification trees that achieve better performance on unseen data (for the tki dataset) than the basic greedy classification trees developed in the first part of the assignment.

Implement your improved models as BetterTree* classes with the same interface as Tree (same build and predict methods). Feel free to try and implement multiple approaches.

For evaluation, use 10-fold cross-validation on the combined -train and -test datasets.

Some approaches you can start with are sensible variable selection, additional pruning, or making the build-process less greedy.

In the report (and during the defense) you must convince your assistant why your improvements can work well and what their trade-offs are.

The improvements will be graded on effectiveness (better performance for the tki dataset), (perceived) generalization to similar datasets of the same type, elegance, and justification. And finally, well-argued negative results can also be acceptable.

[GENERAL NOTES] Your code must be Python 3.12 compatible and must conform to the unit tests from test_hw_tree.py; see tests for the precise interface. In your code, execute anything only under if __name__ == "__main__". Keep the code simple and efficient. Do not explicitly use multithreading/multiprocessing for speed.

You need to write the crux of the solution yourself, but feel free to use libraries for data reading and management (numpy, pandas) and drawing (matplotlib). Submit your code in a single file named hw_tree.py.

Submit a report in a single .pdf file (max two pages). Structure it so your assistant can quickly find the parts he explicitly requested.

## My Results

For this HW, I got an 8- (TA's way of grading). There were two major and one minor problem with the HW:

1. I forgot to include tests into the main `hw_tree.py` file when turning in the assignment.
   a. Although I tested the gini method and split method, the tests were not deep enough → multicolumn split.

2. The importance function was correct and mathematically sound, but the report I turned in didn't correctly display the graph or the usage (the comparison) it should've shown a comparison between two methods (to some baseline) using the importance function.
3. The BetterTree approached were valid, but yielded almost zero improvemnt -> I did get extra points for this, but not enought to counter the previous two problems.
