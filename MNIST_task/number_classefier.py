import matplotlib.pyplot as plt
import seaborn as sns

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
    def find_prediction_indices(self,prediction_list, classification_list):
        index_list = []

        for item in prediction_list:
            for index in range(len(classification_list)):
                if(classification_list[index][1]==item[0]) and (classification_list[index][2]==item[1]):
                    if (classification_list[index]) not in index_list:
                        index_list.append(classification_list[index])
                        break

        return index_list
        

    def print_progress(self, working_index, last_progress_update, update_frequency):
        current_progress = working_index / self.metadataFrame['num_test'].values[0]*100
        new_milestone = None

        if ( int(current_progress) >= int(last_progress_update) ):

            new_milestone = int(current_progress) + update_frequency
            print(f'Progress update: {new_milestone} % ')
        
            return new_milestone
        
        return last_progress_update

    #passing on this for now, might need later
    def divide_dataset_into_chuncks(self, chunk_size):
        pass

    def cluster_data(self):
        print("Started clustering...")
        starting_time = time.time()

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
        self.templates = np.vstack(templates)
        self.template_label = np.array(templatesLabels)

        endtime = time.time()
        duration = endtime-starting_time

        print(f'Finised clustering. \nDuration = {duration}')
        return duration

    #--------Prediction------------------

    def KNN_predict(self, working_node, k_neighbors=1, slow = True):

        #Calculate euclidian distance
        if(slow):
            distance = sp.spatial.distance.cdist(self.train_dataFrame.iloc[:,1:].values,working_node[1:].reshape(1,-1), metric='euclidean').flatten()
        else:
            distance = sp.spatial.distance.cdist(self.templates,working_node[1:].reshape(1,-1), metric='euclidean').flatten()

        #pick out the k nearest neighbours and add their label to KNN_list
        knn_index = np.argsort(distance)[:k_neighbors]

        if(slow):
            knn_lables = [self.train_dataFrame.iloc[index,0] for index in knn_index]
        else:
            knn_lables = [self.template_label[index] for index in knn_index]
        
        #majorety vote
        prediction =  Counter(knn_lables).most_common(1)[0][0]

        return prediction
            
#-------------executions-----------------------
    def run_KNN(self, k_neighbors=1, slow = True, print_progress_updates = False):
        failed_predictions = []
        successfull_predictions = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int) 

        progress = -1
        cluster_duration = 0
        

        if (not slow):
            print(f'Started faster K-neares neighbor classification with k = {k_neighbors}')
            cluster_duration = self.cluster_data()
        else:
            print(f'Started slow K-neares neighbor classification with k = {k_neighbors}')


        starting_time = time.time()
                    
        for working_node_index in range(self.metadataFrame['num_test'].values[0]):

            working_node = self.test_dataFrame.iloc[working_node_index].values
            prediction = self.KNN_predict(working_node, k_neighbors, slow)
            
            correct_label = working_node[0]

            metadata = [working_node_index, prediction, correct_label]

            if (prediction != correct_label):
                failed_predictions.append(metadata)
            else:
                successfull_predictions.append(metadata)

            self.confusion_matrix[correct_label, prediction] += 1

            if(print_progress_updates):
                progress = self.print_progress(working_node_index,progress,1)


        error_rate = len(failed_predictions)/self.metadataFrame['num_test'].values[0]
        endtime = time.time()

        classification_time = endtime-starting_time
        print(f'Finised classification. Error rate: {error_rate} \n Classification duration = {classification_time} \n Total duratiom(with clustering): { cluster_duration+classification_time} ')

        return [classification_time,cluster_duration],error_rate, failed_predictions, successfull_predictions

#-----------------Plotting -----------------------------
    def plot_images(self,image_list, plot_title, fignum = None): 
        sub_fig_lables = ['(a)', '(b)', '(c)', '(d)']
        fig, ax = plt.subplots(2,2, num = fignum)
        fig.suptitle(plot_title, fontsize=16)

        row = 0
        for image_index in range(len(image_list)):
            idx = image_index%2
            if image_index >= 2:
                row =1

            image_data_plot = self.test_dataFrame.iloc[image_list[image_index][0]].values
        
            image_to_plot = image_data_plot[1:].reshape((28, 28))
            correct_label = image_data_plot[0]

            ax[row,idx].imshow(image_to_plot, cmap='gray') 
            ax[row,idx].set_title(f'{sub_fig_lables[image_index]} Correct: {correct_label}, Predcited: {image_list[image_index][1]} ')
            ax[row, idx].axis('off')
        plt.tight_layout()
        #To prevent blocking, plt.show is not called here. It has to be called where the call to plot is located
       
        return fig

    def plot_confusion_matrix(self, plot_title, class_labels, PCR = False, fignum = None):
        fig, ax = plt.subplots(num = fignum)
        fmt="d"

        xlabel = "Correct class"
        ylable = "Predcited class"

        if(PCR):
            self.confusion_matrix = (self.confusion_matrix) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
            fmt = ".1f"
            xlabel += ' [%]'
            ylable += ' [%]'

        sns.heatmap(self.confusion_matrix, annot=True, fmt= fmt, cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels, cbar=True, ax=ax)
        ax.set_title(plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylable)

        return fig
        
        

# klassefier = MNIST_Classefier('MNIST_task/NMIST_data_sets/data_all.mat', 10, [0,1,2,3,4,5,6,7,8,9], 64, 10000)

# print(klassefier.metadataFrame['num_test'].values[0])
# image_list = [[0, 6],
#               [1,4],
#               [2,2]]
              
# klassefier.plot_images(image_list, "some plots")
# # ]

