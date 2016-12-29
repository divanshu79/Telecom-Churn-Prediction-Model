# Telecom-Churn-Prediction-Model

##Background
The goal of this project was to identify which customers would unsubscribe (churn) from a regional cell phone company. The telecom industry has been an area with a large amount of analytic research dedicated to customer retention, due to the large amounts of data generated and the saturation of the marketplace. Customer retention is five to six times cheaper than attracting new customers, and a one percent increase in retention can raise share prices by five percent.

Our data was sourced from a regional cell phone company located in the Southeastern United States, and contains customer information from 1996 through 1999.  The overall churn rate for the customer base was 37%. Certain demographics had notably higher or lower churn rates. For instance, customers far from the southeastern region churned at very high rates (70% +).


###Exploratory analysis
Through visualizations, a few trends were identified in the data:
      -  Customers are from all fifty states, however when looking at the count of customers by region, the company is almost exclusively          regional, as most customers are located in the southeastern portion of the country. 
      -  Churn rates are much higher in regions that are outside southeastern market
      -  Customers who purchased their phones/service plans at mall kiosks or over the internet are mostl likely to churn
      -  Customers who have medium-to-low rate plan are most likely to churn

rates of over 50%

### Predictive Model Building
Several different data mining algorithms were applied to find which model would yield the greatest prediction accuracy.  Algorithms used include decision trees, random forest, support vector machine, adaboost, and neural network. The best model for predicting churn was a random forest, which takes the concept of a decision tree and improves upon it by creating dozens (or hundreds) of possible decision trees, then aggregating them to find the best fit. This model yielded 83% accuracy.

### Results Validation 
Evaluation metrics such as crossvalidation. confusion matrix, ROC curve were used to compare results of each algorithmn. They have proven that random forest is the best algorithm for the telecom dataset.  Though model accuracy is the most important goal of our project, we also find random forest to be an easy technique to work with.  The fact that the model is based on decision tree has made it easier to interpret to upper management. Also, it runs efficiently on our dataset because random forest requires very little data transformation and works very well with categorical data. 


