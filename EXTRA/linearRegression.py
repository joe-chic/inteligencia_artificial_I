from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit() #(x,y) calculates the optimal weights b0 and b1. Takes x features and y target values as inputs, and then solves the OLS equation to find the best fitting line.
# It minimizes the RSS to ensure the predictions are done as close as possible to the actual y-train values.
# The results are stored in coef_ and intercept_.

# X is the design (feature) matrix of size n x p. n is the number of observations and p is the number of features. X^T * X is a square matrix of size p x p. Each entry between
# dot products between two columns in X.

# Why is the Gram matrix ? It contains covariance information about the features. If the columns in the X are orthogonal, the X^T * X is diagonal, meaning the features are independent.
# X^T * X is singular (non-invertible), it means that some features are linearly dependent, causing numerical issues in OLS.

# What does projection mean in linear algebra?  
# How much each feature (column of X) correlates with Y.