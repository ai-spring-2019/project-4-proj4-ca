"""
Name: Caius Arias 
Class: CPSCI 375
Professor: Helmuth

Usage: python3 project3.py DATASET.csv
References Used: Welch Labs - https://www.youtube.com/watch?v=UJwK6jAStmg
"""

import csv, sys, random, math
from numpy import dot, array

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNetwork():
    """Neural network class"""
    def __init__(self, size):
        self.w_in = []
        self.w_out = []
        self.outputs = []
        
        for _ in range(size[0] + 1):
            weights = []
            for _ in range(size[1]):
                weights.append(random.random())
            self.w_in.append(weights)

        #make weights into ndarray and remove w0 for dummy weights from array
        self.w_in = array(self.w_in)
        self.w_0 = self.w_in[-1, :]
        self.w_in = self.w_in[:-1, :]
        
        for _ in range(size[1]):
            weights = []
            for _ in range(size[2]):
                weights.append(random.random())
            self.w_out.append(weights)
        self.w_out = array(self.w_out)
            


    def get_outputs(self):
        return self.outputs


    def forward_propagate(self, training):
        inputs = array([x[1:] for (x, y) in training])

        #multiply w0 by dummy weights
        self.w_0 = training[0][0][0] * self.w_0
        activation = logistic(dot(inputs, self.w_in))

        #for every column j in matrix, add w0,j
        activation = self.w_0 + activation
        
        self.outputs = logistic(dot(activation, self.w_out))
        print(self.outputs)
        
        



        
##
##
##    def predict_class(self):
##
##
##    def back_propagation_learning(self, inputs):



    



def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3])
    nn.forward_propagate(training)
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
