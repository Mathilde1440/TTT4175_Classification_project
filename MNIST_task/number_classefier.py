import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
import time

from collections import Counter



class MNIST_Classefier:

    def __init__(self, file_path, num_classes, class_labels, n_clusters, chunk_size): 

        #---------Initialize member variables-----------
        self.data_filepath = file_path
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.n_clusters = n_clusters

        self.chunk_size = chunk_size

        self.templates = None
        self.template_label = None

        self.confusion_matrix = None

        #load data from .mat file and transform to data frame
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

    def cluster_data(self):
        print("Started clustering...")
        starting_time = time.Time()

        templates = []
        templatesLabels = []

        #Filter out a list of the different classes
        classes = self.train_dataFrame.iloc[:, 0].unique()

        for digit in classes:
            #filter out all data beloning to teh spesific class
            datafilter = self.train_dataFrame.iloc[:, 0] == digit
            relevant_training_data = self.train_dataFrame.iloc[datafilter].iloc[:, 1:].values

            #create cluster
            clusters = sklearn.cluster.KMeans(n_clusters=self.n_clusters, random_state=42)
            id_xi = clusters.fit_predict(relevant_training_data)
            Ci = clusters.cluster_centers_

            #add templates to list
            templates.append(Ci)
            templatesLabels.extend([digit]*self.n_clusters)

        #compress lists and update internal variables
        self.templates = np.vstac(templates)
        self.template_label = np.array(templatesLabels)

        endtime = time.Time()
        duration = endtime-starting_time

        print(f'Finised clustering. \n Duration = {duration}')
        return duration

    #--------Prediction------------------

    def KNN_predict(self, working_node, working_node_index, k_neighbors=1):
        distance = []
        knn_lables = None

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
            
    def fast_KNN_perdict(self, working_node, k_neighbors):

        distance = sp.spatial.distance.cdist(self.templates,working_node[1:].reshape(1,-1), metric='euclidean').flatten()

        #pick out the k nearest neighbours and add their label to KNN_list
        knn_index = np.argsort(distance)[:k_neighbors]
        knn_lables = [self.template_label[index] for index in knn_index]
        
        #majorety vote
        prediction =  Counter(knn_lables).most_common(1)[0][0]

        return prediction
    
#-------------executions-----------------------
    def run_KNN(self, k_neighbors=1):
        total_predictions = []
        failed_prediction_index_list = []
        successfull_prediction_index_list = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int) 

        print(f'Started slow K-neares neighbor classification with k = {k_neighbors}')
        starting_time = time.Time()

        for working_node_index in range(self.metadataFrame['num_test']):

            working_node = self.test_dataFrame.iloc[working_node_index].values

            prediction = self.KNN_predict(working_node, working_node_index, k_neighbors)
            correct_label = prediction != working_node[0]

            if (prediction != correct_label):
                failed_prediction_index_list.append(working_node_index)
            else:
                successfull_prediction_index_list.append(working_node_index)

            total_predictions.append(prediction)
            self.confusion_matrix[correct_label, prediction] += 1

        endtime = time.Time()
        print(f'Finised classification. \n Duration = {endtime-starting_time}')

        return total_predictions, failed_prediction_index_list, successfull_prediction_index_list
    

    def run_KNN_faster (self, k_neighbors=1):
        total_predictions = []
        failed_prediction_index_list = []
        successfull_prediction_index_list = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int) 

        clusterTimeDuration = self.cluster_data()

        print(f'Started faster K-neares neighbor classification with k = {k_neighbors}')
        starting_time = time.Time()

        for working_node_index in range(self.metadataFrame['num_test']):
            working_node = self.test_dataFrame.iloc[working_node_index].values

            prediction = self.fast_KNN_perdict(working_node, k_neighbors)
            correct_label = prediction != working_node[0]

            if (prediction != correct_label):
                failed_prediction_index_list.append(working_node_index)
            else:
                successfull_prediction_index_list.append(working_node_index)

            total_predictions.append(prediction)
            self.confusion_matrix[correct_label, prediction] += 1

        endtime = time.Time()
        classification_time = endtime-starting_time
        print(f'Finised classification. \n Classification duration = {classification_time} \n Total duratiom(with clustering): { clusterTimeDuration+classification_time} ')

        return total_predictions, failed_prediction_index_list, successfull_prediction_index_list


#-----------------Plotting -----------------------------
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

        plt.tight_layout()
        plt.show() 
        
        

# MNist_klassefier = MNIST_Classefier('MNIST_task/NMIST_data_sets/data_all.mat', 10, [0,1,2,3,4,5,6,7,8,9],  10000)
