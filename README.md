# iris
## Description
This is a simple example of a machine learning model that predicts the species of an iris flower based on the length and width of its petals and sepals. The model is trained on the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) from the UCI Machine Learning Repository.

## Usage
### Using Logistic Regression
A logistic regression model is trained on the Iris dataset and used to predict the species of an iris flower based on the length and width of its petals and sepals.
Here, instead of using built-in model of sklearn, we will use our own implementation of logistic regression model.
```python
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def reglrCost(theta, X, y, lambda_s = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    reg = (lambda_s/(2 * m)) * np.sum(theta**2)
    J = J + reg

    return J
def rg(theta, X, y, lambda_s = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    reg = lambda_s * theta /m
    gd = ((1 / m) * X.T.dot(h - y))
    gd = gd + reg

    return gd
def logisticRegression(X, y, theta):
    result = op.minimize(fun = reglrCost, x0 = theta, args = (X, y),
                         method = 'TNC', jac = rg)

    return result.x
```
This is the logistic regression model.
This code is an implementation of logistic regression using regularization to prevent overfitting.

The function sigmoid(z) computes the sigmoid function, which maps any real-valued number to a value between 0 and 1. This function is used to calculate the probability that an input example belongs to the positive class in binary classification.

The function reglrCostFunction(theta, X, y, lambda_s) computes the cost function for logistic regression with regularization. The inputs are the parameters theta, the feature matrix X, the target vector y, and the regularization parameter lambda_s. The cost function is calculated as the negative log-likelihood of the training data plus a regularization term. The regularization term is used to shrink the parameters towards zero, which helps to prevent overfitting. The cost function is minimized using the TNC optimization algorithm.

The function reglrGradient(theta, X, y, lambda_s) computes the gradient of the cost function with respect to the parameters theta. The gradient is calculated as the derivative of the cost function with respect to theta, plus a regularization term. The gradient is used by the optimization algorithm to update the parameters.

The function logisticRegression(X, y, theta) is the main function that performs logistic regression with regularization. It takes as input the feature matrix X, the target vector y, and the initial parameters theta. It calls the op.minimize function to minimize the cost function using the TNC algorithm, and returns the optimized parameters.

Overall, this code can be used to train a logistic regression model for binary classification with regularization.
We have used TNC as optimizer as it is a good optimizer for large datasets.
Then after testing the model we plot the confusion matrix to see the accuracy of the model.

### Using Artificial Neural Network
An artificial neural network is trained on the Iris dataset and used to predict the species of an iris flower based on the length and width of its petals and sepals.
Here, we create a model using keras MLP layers helping the model to learn features
```python
model=Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```
This is description of the model.We have used Dropout layers to prevent overfitting.
We use relu activation function for hidden layers and softmax for output layer. We use categorical crossentropy as loss function and adam as optimizer.
We have used categorical crossentropy as we have 3 classes and adam as it is a good optimizer for large datasets.
As adam allows converges faster than other optimizers like sgd(Stochastic Gradient Descent).

Then after testing the model we plot the confusion matrix to see the accuracy of the model.
