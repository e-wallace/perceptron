'''
Queen's University, Fall 2020
Course: COGS 400 - Neural and Genetic Computing
Assignment 1 - Perceptron Part B - Implement Pocket Algorithm
Student Number: ********
'''
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tqdm import tqdm

def readData():
    #training dataset separated into X_train for features: Sepal length, Sepal width, Petal length, Petal width
    #and y_train for the iris types or labels: Setosa, versicolor, virginica
    X_train = []
    y_train = []
    f = open("iris_train.txt", 'r')
    for line in f:
        line = line.rstrip().split(',')
        X_train.append(np.array(line[:-1], dtype=np.float32))
        y_train.append(line[-1])

    #same is done with the test data
    X_test = []
    y_test = []
    f = open("iris_test.txt", 'r')
    for line in f:
        line = line.rstrip().split(',')
        X_test.append(np.array(line[:-1], dtype=np.float32))
        y_test.append(line[-1])
    return X_train, y_train, X_test, y_test

class Perceptron(object):
    def __init__(self, num_inputs, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros((5, 3))

    def label_to_vector(self, label):
        #one hot encode iris variety
        #array has unique code for each
        if label == "Iris-setosa":
            return np.array([1, 0, 0], dtype=np.float32)
        elif label == "Iris-versicolor":
            return np.array([0, 1, 0], dtype=np.float32)
        elif label == "Iris-virginica":
            return np.array([0, 0, 1], dtype=np.float32)
    
    def vector_to_label(self, vec):
        #reverse of above function
        #allows the label variable to be both a vector and a written label depending on context
        if np.array_equal(vec, [1, 0, 0]):
            return "Iris-setosa"
        elif np.array_equal(vec, [0, 1, 0]):
            return "Iris-versicolor"
        elif np.array_equal(vec, [0, 0, 1]):
            return "Iris-virginica"

    def error(self, label, prediction):
        label_vec = self.label_to_vector(label)
        error = []
        for _ in range(len(label_vec)):
            #error is calculated by actual - predicted iris label
            error.append(label_vec[_]-prediction[_]) 
        return error
    
    def adjust_weights(self, error, inputs):
        for _ in range(len(error)):
            for i, row in enumerate(self.weights):
                #update rule
                row[_] += self.learning_rate * error[_] * inputs[i]

    def predict(self, inputs):
    #vector calculation from weights and input
        net = np.dot(inputs, self.weights)
        largest_sum = -1
        prediction = None
        #three vals in net for the ouput sum at each percept
        for i, sum in enumerate(net):
            if sum > largest_sum:
                largest_sum = sum
                prediction = i
        #create a prediction vector
        p_vec = np.zeros(3)
        p_vec[prediction] = 1
        #fired/ not fired represented by the binary code of the iris label (same in vector_to_label and label_to_vector)
        return p_vec

    def train(self, X_train, labels):
        #pocket alg
        pocket = self.weights
        best_run = 0
        current_run = 0
        for _ in tqdm(range(self.epochs)):
            for inputs, label in zip(X_train, labels):
                # add x0 to the input vector
                inputs = np.insert(inputs, 0, 1)
                # predict and adjust weights
                prediction = self.predict(inputs)
                error = self.error(label, prediction)
                # check the error and update the pocket
                if np.array_equal(error, [0, 0, 0]):
                    current_run += 1
                else:
                    if current_run > best_run:
                        best_run = current_run
                        pocket = self.weights
                    current_run = 0
                self.adjust_weights(error, inputs)
        
        # after all training, set the weight equal to what's in the pocket
        self.weights = pocket

    def test(self, X_test, labels):
        # open a file to write predictions to
        output_file = open("predictions_perceptron.txt", 'w')
        predictions = []
    
        correct_predictions = 0.0
        # create one iterable from inputs and labels
        for inputs, label in zip(X_test, labels):
        #add var to input vector
            inputs = np.insert(inputs, 0, 1)
            prediction = self.predict(inputs)
            output_file.write(f"{str(inputs)},{self.vector_to_label(prediction)}\n")
            predictions.append(self.vector_to_label(prediction))
            #compare predicted to actual 
            desired_output = self.label_to_vector(label)

            #if prediction and actual label match the prediction was correct! woohoo! we add that to the running tally
            if np.array_equal(prediction, desired_output):
                correct_predictions += 1.0
    
        output_file.close()
        
        print(f"Number of Inputs: {len(X_test)}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy of Model: {correct_predictions / float(len(labels))}%\n")
        return predictions

def analysis(y_test, perceptron_predictions):
    #model analysis using sklearn metrics
    conf_matrix= confusion_matrix(y_test, perceptron_predictions, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    precision = precision_score(y_test, perceptron_predictions,labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"], average='micro')   
    recall = recall_score(y_test, perceptron_predictions, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"], average='micro')

    print(conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def main():
    #fill data vars with provided txt files
    X_train, y_train, X_test, y_test = readData()

    #instantiate the perceptron class
    #num inputs: sepal length, Sepal width, Petal length, Petal width
    #num_inputs = 4, epochs = 500, learning_rate = 0.2
    perceptron = Perceptron(4, 500, 0.2)

    #train the perceptron!
    perceptron.train(X_train, y_train)

    #test the perceptron!
    perceptron_predictions = perceptron.test(X_test, y_test)
    
    #show stats
    analysis(y_test, perceptron_predictions)
   

if __name__ == "__main__":
    main()

