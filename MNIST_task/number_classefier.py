import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn


class MNIST_Classefier:

    def __init__(self,training_set, test_set):

        #---------Initialize member variables-----------
        self.training_set = training_set # might actually inculde a built in data module here, we'll see 
        self.test_set = test_set
        

    #---------Util member functions-----------------

    def divide_daset_into_chuncks(self):
        pass

    #--------Traning functions, classefiers and plotting functions-------

    def NN_classefier(self):
        pass


    #Have not completly decided how this one will work
    #Right now i want this to be able to plot a arbitrry number of images, so it can be used 
    #to solve the task relating to plotting both misclasefied and correctly classefied images
    #Now it just plots a couple of pictures in a row, might change to plot a picture grid instead

    #added functions from tips in the task, not implemented correctly
    def plot_images(self, image_list): 

        for image_index in range(len(image_list)):

            image_to_plot = np.testv[image_index, :].reshape((28, 28))
            plt.imshow(image_to_plot, cmap='gray') 

            template = None
            test = None

            distance = sp.spatial.distance.cdist(template,test, metric='euclidean')

            plt.show() 
        

    #added functions from tips in the task, not implemented correctly
    def cluster_train(self):

        train_vi = None
        M = None

        kmeans = sklearn.cluster.KMeans(n_clusters=M, random_state=42)
        id_xi = sklearn.cluster.KMeans.fit_predict(train_vi)
        Ci = sklearn.cluster.KMeans.cluster_centers_

    pass

        





