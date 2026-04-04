# HW2

## Instructions

**Part 1 (Grade 6)**
This part focuses on the attached basketball shot dataset. Our goal is to predict ShotType using all the other variables. You may assume that the dataset is a representative sample of the data generating process. Optimal performance is not important, so no need for data transformations, feature selection, and you can rely on default parameters, unless otherwise noted.

We want to compare 3 models:

- Baseline classifier that only learns and predicts the relative frequencies of classes,
- Logistic regression,
- A model of your choice (it has to be a model whose performance is known to be very sensitive to the choice of at least one parameter).

For the model of your choice you will have to select the tunable parameters for each training fold in cross-validation. Do this in two different ways (we can count these as two separate models):

- Optimizing training fold performance.
- Nested cross-validation.

Our metrics of choice are log-score and classification accuracy.
Implement a model evaluation and comparison of these four models and two metrics using cross-validation. Report the results and your interpretation. Include and motivate any further methodology choices you had to make.

**Part 2 (Grades 7–8)**
*(will only be graded if you successfully submit and defend Part 1)*
After performing the analysis in Part 1 you get two additional requests.
First, we suspect that error depends on Angle. Provide further results that confirm (or disprove) our suspicions. Describe and motivate your methodology.

Second, it turns out that the dataset is not entirely representative of the data generating process – the difference is that the true relative frequencies of Competition types are 0.6 for NBA and 0.1 for the other four types (instead of the approximately equal representation of types in the given dataset). Estimate how the models would perform on
data with the true relative frequencies of Competition type. Report the results and your interpretation. Describe and motivate your methodology.

*Hint*: You don’t have to re-do Part 1. Part 2 can be done well enough by analyzing the errors data from Part 1.

**Part 3 (Grade 9–10)**
*(will only be graded if you successfully submit and defend Parts 1 and 2)*

This part focuses on the paper: Mahoney, Michael J., et al. ”Assessing the performance of spatial cross-validation approaches for models of spatially structured data.” arXiv preprint arXiv:2303.07334 (2023). (link: <https://arxiv.org/pdf/2303.07334>)

Study the paper (and, if necessary, cited or related literature) and prepare yourself for a discussion of different cross-validation strategies for spatial data. In particular, but not limited to:

- Why is model evaluation on spatial data an additional challenge (contrast with IID
and temporal data)?
- How do the methods described in the paper work? What are their advantages and
disadvantages?
- Is any method typically empirically better/worse than the rest?
- Are there any popular approaches not described in the paper?

**General Notes**

- Part 1 can be done in Python or R. Part 2 must be done in R (unless you are not enrolled in the Data Science Track – then you can also use Python for Part 2).
- Submit a pdf report (all parts combined into a single pdf; no more than 1 page per part!) and easy-to-reproduce code.
- The evaluation process (CV, block bootstrap, nested CV, etc.) and evaluation metrics (log-score, MSE, etc.) should be your own code (= don’t rely on autoML-like libraries). For everything else you are encouraged to use existing libraries.
- Feel free to use any tools, including LLMs and collaboration with others, but keep in mind that our goal is to understand what we are doing and not merely to do.

Your work will be graded based on your understanding of your code, report, and the subject matter in general.

## My Results

For this homework I got a 10. We discussed the following:

- What was my model of choice (which were the random forest classifiers), and what parameter did I pick.
- How did I "tune" that parameter using the two ways provided, basically described my approach.
- What did I do for the second part where the data was not representative, how did I weigh and then average (basically what did I do and how does it make sense).

For part 2 I had a big problem that I only showed some correlation between the angle and error, however that is not rigorously correlated since I didn't use some sort of a statistical test to confirm the hypothesis (did get some point deduction here).

For the last part I studied the given paper, described the approached and the "meta" behind them, how now that the data is not i.i.d. we cannot use random folds like in the standard CV, rather we have to take into account the autospatialcorrelation and it's effcts. I also mentioned the paper from [Wadoux et al.](https://www.researchgate.net/publication/353729299_Spatial_cross-validation_is_not_the_right_way_to_evaluate_map_accuracy), which argues that random CV is okay, but measures a different thing than spatial CV does. The prof. seemed to be impressed that I found this and talked more about the "meta" of the approaches instead of the actual methods and implementations.
