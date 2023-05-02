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
def reglrCostFunction(theta, X, y, lambda_s = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    reg = (lambda_s/(2 * m)) * np.sum(theta**2)
    J = J + reg

    return J
def reglrGradient(theta, X, y, lambda_s = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    reg = lambda_s * theta /m
    gd = ((1 / m) * X.T.dot(h - y))
    gd = gd + reg

    return gd
def logisticRegression(X, y, theta):
    result = op.minimize(fun = reglrCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = reglrGradient)

    return result.x
```
This is the logistic regression model.
We have used sigmoid function as activation function.
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
