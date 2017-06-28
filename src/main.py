import numpy as np
import math
import operator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class HypotesisTerm():
    def __init__(self, weight, stump_column, stump_function):
        self.weight = weight
        self.stump_column = stump_column
        self.stump_function = stump_function

    def evaluate_term(self, evaluate_array):
        return self.weight * self.stump_function(evaluate_array[self.stump_column])


class AdaBoostClassifier():
    def __init__(self,maximum_iterations):
        self.maximum_iterations = maximum_iterations
        self.model_terms = []

    def fit(self, X, y):
        self.__generate_stump_combinations(X)
        self.__generate_value_weights(X)
        for iteration in xrange(self.maximum_iterations):
            self.__iterate_training(X, y)

    def fit_with_training_error(self, X, y):
        self.fit(X,y)
        return self.predict_array(X)

    def __generate_stump_combinations(self, X):
        self.stump_combinations = [(column,column_value) for column in range(X.shape[1]) for column_value in range(3)]

    def __generate_value_weights(self, X):
        line, columns = X.shape
        self.value_weights = np.ones((line)) * 1.0/float(line)

    def __iterate_training(self, X, y):
        
        min_weighted_error = float("Inf")
        selected_function = None
        column_selected = None
        x_stump_column_selected = None

        for stump_column, stump_column_value in self.stump_combinations:
            stump_functions = self.__generate_stump_functions(stump_column_value)
            x_stump_column = X[:, stump_column]
            weighted_error, choosen_stump_function = self.__get_weighted_error(x_stump_column, y, stump_functions)
        
            if weighted_error < min_weighted_error:
                min_weighted_error = weighted_error
                selected_function = choosen_stump_function
                column_selected = stump_column
                x_stump_column_selected = x_stump_column

        term_weight = self.__append_model_term(min_weighted_error, column_selected, selected_function)
        self.__update_value_weights(term_weight, y, selected_function, x_stump_column_selected)

    def __generate_stump_functions(self, value_condiction):
        positive_stump = lambda x: 1 if x == value_condiction else -1
        negative_stump = lambda x: 1 if x != value_condiction else -1
        return(positive_stump, negative_stump)

    def __get_weighted_error(self, x_stump_column, y, stump_functions):
        min_weighted_error = float("Inf")
        choosen_stump_function = None
        for function_number, stump_function in enumerate(stump_functions):
            weighted_error = 0.0
            for index, x_value in enumerate(x_stump_column):
                if stump_function(x_value) != y[index]:
                    weighted_error += self.value_weights[index]
            weighted_error = weighted_error/sum(self.value_weights)
            if weighted_error < min_weighted_error:
                min_weighted_error = weighted_error
                choosen_stump_function = stump_function

        return min_weighted_error, choosen_stump_function
    
    def __append_model_term(self, weighted_error, column, stump_function):
        weight = 0.5 * math.log((1.0 - weighted_error)/(weighted_error))
        self.model_terms.append(HypotesisTerm(weight, column, stump_function))
        return weight

    def __update_value_weights(self, weighted_error, y, stump_function, x_column):

        self.__calculate_value_weights(weighted_error, y, stump_function, x_column)
        self.__normalize_value_weights()

    def __calculate_value_weights(self, weighted_error, y, stump_function, x_column):
        for index, value_weight in enumerate(self.value_weights):
            if stump_function(x_column[index]) == y[index]:
                self.value_weights[index] = value_weight * math.exp(-weighted_error)
            elif stump_function(x_column[index]) != y[index]:
                self.value_weights[index] = value_weight * math.exp(weighted_error)

    def __normalize_value_weights(self):
        value_weights_sum = sum(self.value_weights)
        self.value_weights = map(lambda x: x/value_weights_sum, self.value_weights)

    def predict(self, X):
        return 1 if reduce(operator.add, map(lambda y: y.evaluate_term(X), self.model_terms)) >= 0 else -1

    def predict_array(self, X):
        return [self.predict(x) for x in X]

def parse_data(element):
    if element == 'negative':
        return -1
    if element == 'b':
        return 0
    if element == 'x' or element == 'positive':
        return 1
    if element == 'o':
        return 2

def parse_matrix(data_array):
    return np.vectorize(parse_data)(data_array)

# Read dataset from txt
data = np.genfromtxt("../data/tic-tac-toe.data.txt", delimiter=",", dtype=None)

# Convert data
parsed_data = parse_matrix(data)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

start_iteration = 10
end_iteration = 100

error_test = []
error_training = []
for n_iterations in xrange(start_iteration, end_iteration, 10):
    print("Iteration %d" % n_iterations)
    model = AdaBoostClassifier(n_iterations)

    X = parsed_data[:, :-1]
    y = parsed_data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    training_predictions = model.fit_with_training_error(X_train, y_train)
    error_training.append(1-accuracy_score(y_train, training_predictions))
    y_pred = model.predict_array(X_test)
    error_test.append(1-accuracy_score(y_test, y_pred))

plt.plot(error_training)
plt.plot(error_test)
plt.xticks(np.arange(start_iteration, end_iteration+1, 1.0))

plt.legend(['Trainging Error', 'Test Error'], loc='upper left')
plt.show()
