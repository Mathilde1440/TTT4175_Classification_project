import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LCD_training_model:

#-------------Constructor ---------------
    def __init__(self, training_set, testing_set, number_of_classes, number_of_input_variables, class_labels, alpha, iterations):

#Note: right now, all variables are public might change this, might not
        self.training_set = training_set
        self.testing_set = testing_set
        self.number_of_classes = number_of_classes
        self.number_of_input_variables = number_of_input_variables
        self.class_labels = class_labels
        self.alpha = alpha 
        self.iterations = iterations
        
        #Initialize empty vectorrs for MSE and gradient
        #(Initializiung as vectors allows for storing teh oldvalues instead of writing over at each iteration)
        self.MSE_vector = np.zeros(iterations)
        self.MSE_gradient_vector = np.zeros(iterations)

#Possible idea: crete memberfunctions to generate traning and testing dataset based on the number of datapoints used for testing 
#and traing as an input value, instead of having the entire set as input values
#Alternatively, create a separe class/module to handle the datset generation
        

#-------------Not sure if getters and setters are in order, if they are decalre them here ---------------


#------------- util memeber functions ---------------

    def sigmoid(z):
        return 1 / ( 1 + np.exp(-z) )
    
    def linear_discriminant(wi,x,wio):
        return np.transpose(wi)*x + wio
    
    def MSE(gk,tk):
        return (1/2)*np.transpose(gk-tk) * (gk-tk)
    
    def MSE_gradient(gk, tk, xk):

        MSE_gradient_gk = gk-tk
        gradient_g = np.multiply(gk, 1-gk)
        W_gradient = np.transpose(xk)

        return np.multiply(MSE_gradient_gk,gradient_g) * W_gradient
    
    def new_weight(W_old, MSE_gradient, alpha):
        return W_old - alpha * MSE_gradient
    
#----------member functions for trining and testing---------
    def train():
        pass

    def test():
        pass

    

        
        
