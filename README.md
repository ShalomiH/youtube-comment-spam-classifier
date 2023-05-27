# youtube-comment-spam-classifier

This program identifies whether a comment posted on a YouTube video is either spam or not spam. In a Machine Learning context, this is a classification problem where the goal is to predict if a comment is spam or not based on the features. In this case, the comment itself will be the “features” and the target is the label spam or not spam.

In addition to a baseline algorithm, the program builds the following four models: a Multinomial Naive Bayes model, a k-Nearest Neighbor model, a Support Vector Machine model, and a Decision Tree Classifier model.

These models are tuned according to the model score on the validation test (where we use hold-out validation for the baseline and k-fold validation for the four Machine Learning algorithms) – that is, we use the accuracy score. Additionally, the final model score on the performance of each model on the test set is also a calculation of the accuracy score. The model that has the highest performance is selected to predict whether the new, unseen comments are spam or not spam.



## Running Instructions
Navigate to the terminal directory containing the code.py and YouTube-Spam-Collection-v1.zip files and run:
```
python3 code.py
```
 - Note 1: the program may take a few minutes to run.
 - Note 2: For the libraries and tools used, see “Sources Referenced and Tools Used” below.



## Results and Evaluation:
The full YouTube-Spam-Collection-v1.zip dataset is visualized below:
![image](https://github.com/ShalomiH/youtube-comment-spam-classifier/assets/90998772/e9248a3a-e51d-4c51-b925-79d3e7265648)

The following is a comparison of the resulting models built is a sample run of the program using their cross validation scores:
![image](https://github.com/ShalomiH/youtube-comment-spam-classifier/assets/90998772/6555bc39-e0c0-4590-80b9-789c3d616dd7)

In this example, the performance score of the model selected as the best (in this case, the Decision Tree Classifier model) is approximately 0.9522998.

Upon running this program multiple times, we have found that the Decision Tree Classifier model has consistently out-performed the other models. This Decision Tree Classifier approach was best suited to this Machine Learning classification task and dataset in terms of accuracy.



## Sources Referenced and Tools Used
#### Dataset source: https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
#### Python libraries used: pandas, numpy, zipfile, os, mathplotlib, sklearn.
#### Additional resources consulted:
[1] - https://pandas.pydata.org/docs/user_guide/merging.html

[2] - https://lifewithdata.com/2022/03/23/how-to-create-a-baseline-classification-model-in-scikit-learn/

[3] - https://learn-scikit.oneoffcoder.com/hyperparam-tuning.html

[4] - https://bait509-ubc.github.io/BAIT509/lectures/lecture6.html

[5] - https://stackoverflow.com/questions/33830959/multinomial-naive-bayes-parameter-alpha-setting-scikit-learn

[6] - https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/

[7] - https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

[8] - https://plainenglish.io/blog/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda

[9] - https://vitalflux.com/decision-tree-hyperparameter-tuning-grid-search-example/
