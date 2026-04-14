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

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels,1)
    failure_fig = mnist_klassefier.plot_images(failed_predictions[:4], "Failed predictions",2)
    succsess_fig = mnist_klassefier.plot_images(successfull_predictions[:4], "Successful predictions",3 )
    plt.show()
    plt.close('all')
    
    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        failure_fig.savefig(figure_names[1])
        succsess_fig.savefig(figure_names[2])


        destination_folder = Path('MNIST_task/plots/'+ folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create the folder if it doesn't exist

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))

        with open(os.path.join(destination_folder, 'stats.txt'), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n')
            f.write(f'Total Classification time: {time[0]}\n')
            f.write(f'Cluster time:  {time[1]}\n')
            f.write(f'Classification (wo. clustertime): {time[0]-time[1]}\n')

    return error_rate

def solve_task_two(folder_path=None,figure_names=None, save_figures=False, k_neighbors = 1):

    mnist_klassefier =  MNIST_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    time,error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(k_neighbors = k_neighbors , slow = False)

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels)
    failure_fig = mnist_klassefier.plot_images(failed_predictions[:4], "Failed predictions",2)
    succsess_fig = mnist_klassefier.plot_images(successfull_predictions[:4], "Successful predictions",3 )

    plt.show()
    plt.close()

    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        failure_fig.savefig(figure_names[1])
        succsess_fig.savefig(figure_names[2])

        destination_folder = Path('MNIST_task/plots/'+ folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create the folder if it doesn't exist

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))

        with open(os.path.join(destination_folder, 'stats.txt'), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n')
            f.write(f'Total Classification time: {time[0]}\n')
            f.write(f'Cluster time:  {time[1]}\n')
            f.write(f'Classification (wo. clustertime): {time[0]-time[1]}\n')

    return error_rate

folder_path = 'test'

fig_labels_task1 = ['1_cm_t1.pdf',
                    '1_f_fig_t1.pdf',
                    '1_s_fig_t1.pdf']

fig_labels_task2A = ['t_cm_2A.pdf',
                     't_f_fig_2A.pdf',
                     't_s_fig_2A.pdf']

fig_labels_task2B = ['1_cm_2B.pdf',
                     '1_f_fig_2B.pdf',
                     '1_s_fig_2B.pdf']

error_1 = solve_task_one(folder_path = folder_path ,figure_names=fig_labels_task2A,save_figures=True )
error_2A = solve_task_two(folder_path = folder_path ,figure_names=fig_labels_task2A,save_figures=True )
error_2B = solve_task_two(folder_path = folder_path ,figure_names=fig_labels_task2A,save_figures=True )
