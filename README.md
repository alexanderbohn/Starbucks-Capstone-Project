# Starbucks-Capstone-Project

## Project Overview

In today's customer-centric world, data is becoming more valuable as the insight it provides helps businesses understand details of its customer base. For example, the 360-degree view is the foundation that makes an organization's relationship with customers experiential rather than transactional — the key to long-standing customer relationships. It is the understanding that companies can obtain a comprehensive view of customers by leveraging data from various touchpoints in a customer’s journey.

This dataset contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. The objective is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type.

Every offer has a validity period before the offer expires. For example, a BOGO offer might be valid for only 5 days. In the dataset there are informational offers that have a validity period even though these ads are merely providing information about a product. For example, if an informational offer has 7 days of validity, the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

Additionally, there is transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. There are cases where someone using the app might make a purchase through the app without having received an offer or seen an offer.

## Problem Statement

I am interested in answering the following questions:
* Which offer should be sent to a particular customer to let the customer buy more?
* Which demographic groups respond best to which offer type?

The way in which I am aiming to answer the questions above is shown below:
* __Step 1:__ Install Python libraries and download/retrieve the datasets
* __Step 2:__ Pre-process offer portfolio, customer profile, and transaction data
* __Step 3:__ Combine pre-processed data to illustrate offer attributes, customer demographic, and transaction detail on a record basis
* __Step 4:__ Split the aggregated dataset into training/testing data
* __Step 5:__ Perform data analysis and visualizations for each offer type
* __Step 6:__ Train models and evaluate model performances 
    * Naive Classifier model (assumes all offers were successful)
    * Classification models (Logistic Regression, Random Forest, and Gradient Boosting)
* __Step 7:__ Tune best performing training model and evaluate model on test data

## Metrics

The metrics to quantify the performance of both the benchmark model (Naive Classifier) and the classification models will be the accuracy and the F1 score.

The accuracy measures the fraction of predictions that the model got right and is defined below:

Accuracy = Correct Predictions / Total Predictions

The F1 score is also a measure of a test’s accuracy except that it considers both the precision and the recall to compute the score. Prior to defining the F1 score however, I will need to define the precision and recall. 

Precision describes how precise the model is out of the predicted positive cases and is defined below: 

Precision = True Positive / (True Positive + False Positive)

Recall describes how many cases were actual positives through the model labeling it as positive and is defined below: 

Recall = True Positive / (True Positive + False Negative)

With the precision and recall definitions in mind, the F1 score metric can be interpreted as the weighted average of the precision and recall and is defined below:

F1 score = 2 x ((Precision * Recall) / (Precision + Recall))

## Results

Model ranking based on training data Accuracy
* Random Forest Classifier accuracy: 87.66%
* Gradient Boosting Classifier accuracy: 74.80%
* Logistic Regression accuracy: 66.66%
* Naive predictor accuracy: 52.16%

Model ranking based on training data F1 score
* Random Forest Classifier F1 score: 88.13%
* Gradient Boosting Classifier F1 score: 76.92%
* Logistic Regression F1 score: 74.04%
* Naive predictor F1 score: 68.56%

The analysis from the training model step indicated that the Random Forest model had the best training data accuracy and F1 score with values of 87.66% and 88.13% respectively. The last step involved optimizing a set of the model's hyperparameters using a RandomizedSearch. Using the best parameters obtained from the tuning process, the test data accuracy and F1 score had values of 74.57% and 76.86% respectively. These numbers indicate that the model did not overfit the training data.

From the Feature Importance step, the top five features from the Random Forest model with the training data were:
<ol>
    <li>Income</li>
    <li>Offer Duration</li>
    <li>Informational Offer Type</li>
    <li>Required Minimum Spend</li>
    <li>Customers who created an account in 2018</li>
</ol>

Given how the income as well as the offer duration and how much money a customer must spend to complete an offer are in the list above, it may be possible to improve the Random Forest model's performance. One way to do so is by  incorporating new features that illustrate an offer's success rate, taking into account a customer's income, the offer duration and the required minimum spend.

## Jupyter Notebook Environment 
* Starbucks_Capstone_notebook.ipynb
  * Install Python libraries and download/retrieve the datasets
  * Pre-process offer portfolio, customer profile, and transaction data
  * Combine pre-processed data to illustrate offer attributes, customer demographic, and transaction detail on a record basis
  * Split the aggregated dataset into training/testing data
  * Perform data analysis and visualizations for each offer type
  * Train models and evaluate model performances 
  * Tune best performing training model and evaluate model on test data
* data folder
  * portfolio.json - contains offer ids and metadata about each offer type
  * profile.json - demographic data for each customer
  * transcript.json - records for transactions, offers received, offers viewed, and offers completed
  * clean_data.csv - Combined pre-processed data of encompassing portfolio, profile and transcript data
  
## Python Libraries
* Pandas
* Numpy
* Matplotlib
* Seaborn
* scikit-learn
* os
* re
* progressbar

## References
Starbucks Capstone Challenge Github repo: https://github.com/mspcvsp/StarbucksCapstoneChallenge<br>
The What, Why & How of the 360-Degree Customer View: https://digitalmarketinginstitute.com/en-us/blog/the-what-why-and-how-of-360-degree-customer-view<br>
MultiLabelBinarizer: https://www.kaggle.com/questions-and-answers/66693<br>
Pandas get_dummies function: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html<br>
Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html<br>
Progress Bar documentation: https://progressbar-2.readthedocs.io/en/latest/#<br>
Train/Test Split: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6<br>
Random State in Splitting Dataset: https://stackoverflow.com/questions/42191717/python-random-state-in-splitting-dataset<br>
Feature Scaling: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html<br>
Scikit-learn MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html<br>
Numpy "-1" reshape: https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape<br>
Reverse one-hot-encoding: https://stackoverflow.com/questions/38334296/reversing-one-hot-encoding-in-pandas<br>
Custom sorting pandas df: https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe<br>
Machine Learning Evaulate Classification Model: https://www.ritchieng.com/machine-learning-evaluate-classification-model/<br>
Logistic Regression Detailed Overview: https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc<br>
Understanding Random Forest: https://towardsdatascience.com/understanding-random-forest-58381e0602d2<br>
Random Forest Classifier Example: https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/<br>
Understanding Gradient Boosting Machines: https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab<br>
Understanding Confusion Matrix: https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62<br>
Receiver Operating Characteristics (ROC): https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5<br>
Hyperparameter Tuning Random Forest: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74<br>
How to Save a Model: https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli<br>
