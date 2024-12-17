# Yelp-Recommender
Recommender system project for DSCI 553 - Foundations and Applications of Data Mining at the University of Southern California (USC).

* Goal: Given a user-business pair on Yelp, accurately predict the rating given to the business by the user.
    * Training dataset of 455,854 points
    * Validation dataset of 142,044 points
* Weighted hybrid recommender system with model-based and item-based collaborative filtering components
* Final RMSE: 0.98477 stars
* Execution time: 96 seconds
* Error distribution:

| Error range (stars) | Number of ratings in range |
| ------------ | ------------ |
| \>=0 and <=1 | 101,530 |
| \>1 and <=2 | 33,427 |
| \>2 and <=3 | 6,327 |
| \>3 and <=4 | 760 |
| \>4 and <=5 | 0 |


* Future improvements:
    * Including even more features from dataset: number of Yelp friends a user has, compliments on Yelp profile, etc.
    * Upgrading existing packages (XGBoost, PySpark, Scikit-Learn) to take advantage of latest features
        * Fitting XGBoost model with `reg:squarederror` loss function, learning rate scheduler, dynamic early stopping threshold
        * Dynamically weighting hybrid recommender system or focusing on features to determine weights
