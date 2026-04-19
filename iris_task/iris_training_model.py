import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

#ggs
class LCD_training_model:

#-------------Constructor ---------------
    def __init__(self, training_set, testing_set, number_of_classes, number_of_input_variables, class_labels, alpha, iterations, random_initial_weight_matrix):

        self.training_set = training_set
        self.testing_set = testing_set
        self.number_of_classes = number_of_classes
        self.number_of_input_variables = number_of_input_variables
        self.class_labels = class_labels
        self.alpha = alpha 
        self.iterations = iterations

        #Generate weight matrix. Can either be initialized to zero, or with random values
        if random_initial_weight_matrix:
            self.W = np.random.rand(number_of_input_variables,number_of_classes)

        else:
            self.W = np.zeros((number_of_input_variables, number_of_classes))

        #Initialize empty vectors and variables for MSE and gradient
        #(Initializiung as vectors allows for storing teh oldvalues instead of writing over at each iteration)
        self.MSE_vector = np.zeros(iterations)
        self.MSE_gradient_vector = np.zeros(iterations)

        #initialize other needed variables
        self.target_matrix = np.eye(number_of_classes)

#Possible idea: crete memberfunctions to generate traning and testing dataset based on the number of datapoints used for testing 
#and traing as an input value, instead of having the entire set as input values
#Alternatively, create a separe class/module to handle the datset generation
        
#-------------Not sure if getters and setters are in order, if they are decalre them here ---------------

#------------- util memeber functions ---------------
    
    def sigmoid(self,z):
        return 1 / ( 1 + np.exp(-z) )
    
    # def linear_discriminant(wi,x,wio):
    #     wi_T = np.transpose(wi)
    #     return np.dot(wi_T,x) + wio

    def linear_discriminant(self,x):
        #return np.matmul(self.W.T, x)
        W_T = np.transpose(self.W)
        return np.matmul(W_T,x) #DxC
    
    def MSE(self,gk,tk):

        return (1/2)*np.transpose(gk-tk) @ (gk-tk)
    
    def MSE_gradient(self,gk, tk, xk): # this might be wrong, does it need the outer product??

        MSE_gradient_gk = gk-tk
        gradient_g = gk*(1-gk)  #paramaters might need to be declared in opsite order

        return np.outer(xk, MSE_gradient_gk*gradient_g)
    
    def update_W(self,MSE_gradient):
        self.W = self.W - self.alpha * MSE_gradient
    
    def label_to_target_index(self, label):
        for index in range(len(self.class_labels)):
            if (label == self.class_labels[index]):
                return index
            
    #this is unused and a product of a early misunderstanding
    def safe_accumulative_vector_update(self, index, vector, value):
        if(index == 0):
            vector[index] = value
        else:
            vector[index] = vector[index-1] + value

#----------member functions for training and testing---------

#traning function using a batch traing approach
    def train(self):
        print(f'Started training with alpha = {self.alpha} and {self.iterations} iterations \n')
        print("Progress: ", end = '')
        for index in range(self.iterations):
            MSE_k = 0
            MSE_grad_k = np.zeros_like(self.W)

            for xk, label in self.training_set:

                z = self.linear_discriminant(xk)
                gk = self.sigmoid(z)

                tk = self.target_matrix[self.label_to_target_index(label)]

                MSE_k += self.MSE(gk,tk)
                MSE_grad_k += self.MSE_gradient(gk,tk,xk)



            #update weight matrix
            self.update_W(MSE_grad_k)

            #Update accumulators
            self.MSE_vector[index] = MSE_k
            self.MSE_gradient_vector[index] = np.linalg.norm(MSE_grad_k)

            #print progressbar
            if(index % 100 == 0):
                print(f'#', end='', flush = True)
        print(f'\n')

            
#testing function
    def test(self, dataset):
        correct = 0
        total = len(dataset)
        confusion_matrix = np.zeros((self.number_of_classes, self.number_of_classes), dtype=int)

        for xk, label in dataset:
            z = self.linear_discriminant(xk)
            gk = self.sigmoid(z)

            predicted_index = np.argmax(gk)
            true_index = self.label_to_target_index(label)

            confusion_matrix[true_index, predicted_index] += 1

            if predicted_index == true_index:
                correct += 1

        error_count = total - correct
        error_rate = error_count / total

        return correct, error_count, error_rate, confusion_matrix
    

    def print_results(self, correct, error_count, error_rate,confusion_matrix, test):

        if(test):
            print("Testing set results: ")
        else:
             print("Training set results: ")

        print("-----------------------------------")

        print(f'Correct classifications: {correct}')
        print(f'Misclassifications: {error_count}')
        print(f'Error rate: {round(error_rate,3)} %')
        print("Confusion matrix:")
        print(f'{pd.DataFrame(confusion_matrix, index = self.class_labels,columns= self.class_labels )}')
        print("")
        pass


df = pd.read_csv(
    #"iris_task/iris_data_sets/iris.csv",
    "iris_data_sets/iris.csv",
    header=None
)

class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

class_1 = df.iloc[0:50]
class_2 = df.iloc[50:100]
class_3 = df.iloc[100:150]


def dataframe_to_dataset(dataframe):
    dataset = []

    for _, row in dataframe.iterrows():
        xk = row.iloc[0:4].to_numpy(dtype=float)
        label = row.iloc[4]
        dataset.append((xk, label))

    return dataset
