import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import pandas as pd

from collections import Counter


class MNIST_Classefier:
    #filpath list takes the form ['filepath_training_set', 'filepath_test_set', 'filepath_metadata']

    def __init__(self, file_path, num_classes, class_labels, chunk_size): 

        #---------Initialize member variables-----------

        self.data_filepath = file_path
        self.num_classes = num_classes
        self.class_labels = class_labels

        self.chunk_size = chunk_size

        self.templates = None
        self.temple_label = None

        #load data and transform to data frame
        data = sp.io.loadmat(file_path)

        self.metadataFrame = pd.DataFrame({
            'vec_size': [data['vec_size'].item()],
            'num_train': [data['num_train'].item()],
            'num_test':  [data['num_test'].item()]
             })

        self.train_dataFrame = pd.DataFrame(data['trainv'])
        self.train_dataFrame.insert(0, 'label', data['trainlab'])
        self.test_dataFrame = pd.DataFrame(data['testv'])
        self.test_dataFrame.insert(0, 'label', data['testlab'])



    #---------Util member functions-----------------

    #passing on this for now, might need later
    def divide_dataset_into_chuncks(self, chunk_size):
        pass

    #--------Traning functions, classefiers and plotting functions-------

    def slow_KNN_predict(self, working_node,working_node_index, k_neighbors=1):
        distance = []

        for comp_node_index in range(self.metadataFrame['num_train']):
            node_to_compare = self.train_dataFrame.iloc[comp_node_index].values
            euclidian_distance = np.linalg.norm(node_to_compare[1:] - working_node[1:])

            #stop self prediction and Prevent node from beliving it is its own neares neighbor
            if(working_node_index==comp_node_index):
                euclidian_distance = np.inf 

            distance.append(euclidian_distance)

        #pick out the k nearest neighbours and add their label to KNN_list
        knn_index = np.argsort(distance)[:k_neighbors]
        knn_lables = [self.train_dataFrame.iloc[index,0] for index in knn_index]
        
        #majorety vote
        prediction =  Counter(knn_lables).most_common(1)[0][0]

        return prediction
            

    # def fast_KNN_perdict(self, K=1):
    #     training_features = self.train_dataFrame.iloc[:,1].values
    #     training_labels = self.train_dataFrame.iloc[:,0].values

    #     clusters = sklearn.cluster.KMeans(n_clusters=k_neighbors, random_state=42)
    #     id_xi = clusters.fit_predict(training_features)
    #     Ci = clusters.cluster_centers_

    #     for n_class in range(self.num_classes):

    #         filter = (n_class == training_labels)

    def train_KNN(self, k_neighbors=1):
        total_predictions = []
        failed_prediction_index_list = []
        successfull_prediction_index_list = []
 

        for working_node_index in range(self.metadataFrame['num_train']):
            working_node = self.train_dataFrame.iloc[working_node_index].values

            prediction = self.slow_KNN_predict(working_node, working_node_index, k_neighbors)

            if (prediction != working_node[0]):
                failed_prediction_index_list.append(working_node_index)
            else:
                successfull_prediction_index_list.append(working_node_index)

            total_predictions.append(prediction)

        return total_predictions, failed_prediction_index_list, successfull_prediction_index_list

    def train_NN(self, k_neighbors=1):
        total_predictions = []
        failed_prediction_index_list = []
        successfull_prediction_index_list = []
 

        for working_node_index in range(self.metadataFrame['num_train']):

            distance = []
            KNN_s = []
            working_point = self.train_dataFrame.iloc[working_node_index].values

            for comp_node_index in range(self.metadataFrame['num_train']):

                node_to_compare = self.train_dataFrame.iloc[comp_node_index].values
                euclidian_distance = np.linalg.norm(node_to_compare[1:] - working_point[1:])

                distance.append(euclidian_distance)

            #pick out the k nearest neighbours and add their label to KNN_list
            # for i in range(k_neighbors):
            knn_index = np.argsort(distance)[:k_neighbors]
            knn_lables = [self.train_dataFrame.iloc[index,0].values for index in knn_index]
            

            
            prediction =  Counter(knn_lables).most_common

            if (prediction != working_point[0]):
                failed_prediction_index_list.append[working_node_index]
            else:
                successfull_prediction_index_list.append(working_node_index)

            total_predictions.append(prediction)

        return

    def train_KNN_faster(self, k_neighbors=1):
        pass



        

        # distance = sp.spatial.distance.cdist(template,test, metric='euclidean')
        # distance = np.linalg.norm(template - test)
        


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
    def cluster_data(self, n_clusters):

        templates = []
        templatesLabels = []

        #Filter out a list of teh different classes
        classes = self.train_dataFrame.iloc[:, 0].unique()

        for digit in classes:
            #filter out all data beloning to teh spesific class
            datafilter = test.train_dataFrame.iloc[:, 0] == 4
            relevant_training_data = self.train_dataFrame.iloc[datafilter].values

            #create cluster
            clusters = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42)
            id_xi = clusters.fit_predict(relevant_training_data)
            Ci = clusters.cluster_centers_

            #add templates to list
            templates.append(Ci)
            templatesLabels.extend(digit*n_clusters)

        #compress lists and update internal variables
        self.templates = np.vstac(templates)
        self.temple_label = np.vstac(templatesLabels)



test = MNIST_Classefier('MNIST_task/NMIST_data_sets/data_all.mat', 10, [0,1,2,3,4,5,6,7,8,9],  10000)
# print(test.train_dataFrame)
image_list = [test.train_dataFrame.iloc[0].values,
              test.train_dataFrame.iloc[1].values, 
              test.train_dataFrame.iloc[2].values,
              test.train_dataFrame.iloc[3].values]
# ]
# print(image_list[0][0])

datafilter = ([0,1,2,3,4,5,6,7,8,9] == 4)

print(test.train_dataFrame.iloc[:, 0] == 4)
class_data = test.train_dataFrame[test.train_dataFrame.iloc[:, 0] == 4]
# print(class_data)

# relevant_training_data = test.train_dataFrame.iloc[datafilter,:].values

# print(relevant_training_data)
# test.plot_images(image_list)