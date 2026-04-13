import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import pandas as pd


class MNIST_Classefier:
    #filpath list takes the form ['filepath_training_set', 'filepath_test_set', 'filepath_metadata']

    def __init__(self, file_path, chunk_size): 

        #---------Initialize member variables-----------
        # self.training_set = training_set # might actually inculde a built in data module here, we'll see 
        # self.test_set = test_set
        self.data_filepath = file_path
        self.chunk_size = chunk_size

        #load data and transform to data frame
        data = sp.io.loadmat(file_path)

        self.metadataFrame = pd.DataFrame({
            'vec_size': [data['vec_size'].item()],
            'num_train': [data['num_train'].item()],
            'num_test':  [data['num_test'].item()]
             })
        # self.vec_size = data['vec_size']
        # self.num_train = data['num_train']
        # self.num_test = data['num_test']

        self.train_dataFrame = pd.DataFrame(data['trainv'])
        self.train_dataFrame.insert(0, 'label', data['trainlab'])
        self.test_dataFrame = pd.DataFrame(data['testv'])
        self.test_dataFrame.insert(0, 'label', data['testlab'])



       
        

    #---------Util member functions-----------------

    #passing on this for now, might need later
    def divide_dataset_into_chuncks(self, chunk_size):
        pass

    #--------Traning functions, classefiers and plotting functions-------

    def NN_classefier(self):

        # distance = sp.spatial.distance.cdist(template,test, metric='euclidean')
        pass


    #added functions from tips in the task, not implemented correctly
    #right now this plots based in a list of image arrays
    def plot_images(self, image_list): 

      

        fig, ax = plt.subplots(2,2)
        row = 0
        for image_index in range(len(image_list)):
            idx = image_index%2
            if image_index >= 2:
                row =1

            image_to_plot = image_list[image_index][1:].reshape((28, 28))

            ax[row,idx].imshow(image_to_plot, cmap='gray') 
            ax[row,idx].set_title(f'Label = {image_list[image_index][0]}')

            ax[row, idx].legend()


            # image_to_plot = image_list[image_index]

            # image_to_plot = np.testv[image_index, :].reshape((28, 28))
       

            # template = None
            # test = None

            # distance = sp.spatial.distance.cdist(template,test, metric='euclidean')
        plt.tight_layout()
        plt.show() 
        

    #added functions from tips in the task, not implemented correctly
    def cluster_train(self):

        train_vi = None
        M = None

        kmeans = sklearn.cluster.KMeans(n_clusters=M, random_state=42)
        id_xi = sklearn.cluster.KMeans.fit_predict(train_vi)
        Ci = sklearn.cluster.KMeans.cluster_centers_

    pass


test = MNIST_Classefier('MNIST_task/NMIST_data_sets/data_all.mat', 10000)
# print(test.train_dataFrame)
image_list = [test.train_dataFrame.iloc[0].values,
              test.train_dataFrame.iloc[1].values, 
              test.train_dataFrame.iloc[2].values,
              test.train_dataFrame.iloc[3].values]
# print(image_list)

test.plot_images(image_list)