from number_classefier import MNIST_Classefier
import matplotlib.pyplot as plt

import os
from pathlib import Path

def solve_task_one(folder_path=None,figure_names=None, save_figures=False):

    mnist_klassefier =  MNIST_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    time, error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(slow = True)

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels,fignum=1)
    confusion_matrix_fig_PCR = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels, PCR = True,fignum=2)
    failure_fig = mnist_klassefier.plot_images(failed_predictions[:4], "Failed predictions",3)
    succsess_fig = mnist_klassefier.plot_images(successfull_predictions[:4], "Successful predictions",4 )
    
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')
    
    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        confusion_matrix_fig_PCR.savefig(figure_names[1])
        failure_fig.savefig(figure_names[2])
        succsess_fig.savefig(figure_names[3])

        destination_folder = Path('MNIST_task/plots/'+ folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create the folder if it doesn't exist

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))
        os.replace(figure_names[3], os.path.join(destination_folder, figure_names[3]))


        time_min = int(time[0] / 60)
        time_s = int(time[0] - time_min * 60)

        with open(os.path.join(destination_folder, 'stats_1.txt'), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n s')
            f.write(f'Total Classification time: {time[0]} ~ {time_min} min {time_s} s \n')

    return error_rate

def solve_task_two(folder_path=None,figure_names=None, save_figures=False, k_neighbors = 1):

    mnist_klassefier =  MNIST_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    time,error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(k_neighbors = k_neighbors , slow = False)

    intersting_misclassifications = [[5,3], [3,8], [1, 7], [9,4]]
    intersting_correctclassifications = [[1,1], [0,0], [2,2], [8,8]]

    miscalc_plot = mnist_klassefier.find_prediction_indices(intersting_misclassifications, failed_predictions)
    succsess_plot = mnist_klassefier.find_prediction_indices(intersting_correctclassifications, successfull_predictions)
    #failed_predictions[:4]
    #successfull_predictions[:4]
    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels, fignum=1)
    confusion_matrix_fig_PCR = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels, PCR = True,fignum=2)
    failure_fig = mnist_klassefier.plot_images(miscalc_plot, "Failed predictions",3)
    succsess_fig = mnist_klassefier.plot_images(succsess_plot, "Successful predictions",4 )

    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        confusion_matrix_fig_PCR.savefig(figure_names[1])
        failure_fig.savefig(figure_names[2])
        succsess_fig.savefig(figure_names[3])

        destination_folder = Path('MNIST_task/plots/'+ folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create the folder if it doesn't exist

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))
        os.replace(figure_names[3], os.path.join(destination_folder, figure_names[3]))

        txt_filname = 'stats_2A.txt'
        if (k_neighbors != 1):
            txt_filname = 'stats_2B.txt'
   

        with open(os.path.join(destination_folder, txt_filname), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n')
            f.write(f'Total Classification time: {time[0]}\n s')
            f.write(f'Cluster time:  {time[1]}\n s')
            f.write(f'Classification (wo. clustertime): {time[0]+time[1]} s\n')

    return error_rate

folderPath = 'test_plot_3'

fig_labels_task1 = ['cm_t1.pdf',
                    'cm_PCR_t1.pdf',
                    'f_fig_t1.pdf',
                    's_fig_t1.pdf']

fig_labels_task2A = ['cm_2A.pdf',
                     'cm_PCR_2A.pdf',
                     'f_fig_2A.pdf',
                     's_fig_2A.pdf']

fig_labels_task2B = ['cm_2B.pdf',
                     'cm_PCR_2B.pdf',
                     'f_fig_2B.pdf',
                     's_fig_2B.pdf']

error_1 = solve_task_one(folder_path = folderPath ,figure_names=fig_labels_task1,save_figures=True )
error_2A = solve_task_two(folder_path = folderPath ,figure_names=fig_labels_task2A,save_figures=True)
error_2B = solve_task_two(k_neighbors=7, folder_path = folderPath ,figure_names=fig_labels_task2B,save_figures=True )
