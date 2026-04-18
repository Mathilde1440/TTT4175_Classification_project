from number_classefier import KNN_Classefier
import matplotlib.pyplot as plt

import os
from pathlib import Path

def solve_slow_KNN (k_neighbors=1,
                   folder_path=None,
                   figure_names=None, 
                   txt_filname=None,
                   save_figures=False,
                   save_stats=False):

    mnist_klassefier =  KNN_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    time, error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(k_neighbors= k_neighbors,print_progress_updates=True,slow = True)

    intersting_misclassifications = [[5,3], [3,8], [1, 7], [9,4]]
    intersting_correctclassifications = [[1,1], [0,0], [2,2], [8,8]]
    miscalc_plot = mnist_klassefier.find_prediction_indices(intersting_misclassifications, failed_predictions)
    succsess_plot = mnist_klassefier.find_prediction_indices(intersting_correctclassifications, successfull_predictions)

    #failed_predictions[:4]
    #successfull_predictions[:4]

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels,fignum=1)
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
            os.makedirs(destination_folder)  

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))
        os.replace(figure_names[3], os.path.join(destination_folder, figure_names[3]))

    if(save_stats):
        time_min = int(time[0] / 60)
        time_s = int(time[0] - time_min * 60)

        with open(os.path.join(destination_folder, txt_filname), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n s')
            f.write(f'Total Classification time: {time[0]} ~ {time_min} min {time_s} s \n')

    return error_rate

def solve_fast_KNN (k_neighbors = 1,
                   folder_path=None,
                   figure_names=None,
                   txt_filname=None, 
                   save_figures=False,
                   save_stats=False):

    mnist_klassefier =  KNN_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
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
            os.makedirs(destination_folder)

        os.replace(figure_names[0], os.path.join(destination_folder, figure_names[0]))
        os.replace(figure_names[1], os.path.join(destination_folder, figure_names[1]))
        os.replace(figure_names[2], os.path.join(destination_folder, figure_names[2]))
        os.replace(figure_names[3], os.path.join(destination_folder, figure_names[3]))

    if(save_stats):
        with open(os.path.join(destination_folder, txt_filname), 'w') as f:
            f.write(f'Error rate: {error_rate}\n')
            f.write(f'Failed predictions: {len(failed_predictions)}\n')
            f.write(f'Total Classification time: {time[0]} s\n')
            f.write(f'Cluster time:  {time[1]} s \n')
            f.write(f'Classification (wo. clustertime): {time[0]+time[1]} s\n')

    return error_rate

folderPath = 'new_plots_11' 

txt_filename_1 = 'Stats_slow_NN'
txt_filename_2 = 'Stats_slow_KNN'
txt_filename_3 = 'Stats_fast_NN'
txt_filename_4 = 'Stats_fast_KNN'

fig_labels_slow_NN = ['cm_slow_NN.pdf',
                    'cm_PCR_slow_NN.pdf',
                    'f_fig_slow_NN.pdf',
                    's_fig_slow_NN.pdf']

fig_labels_slow_KNN = ['cm_slow_KNN.pdf',
                     'cm_PCR_slow_KNN.pdf',
                     'f_fig_slow_KNN.pdf',
                     's_fig_slow_KNN.pdf']

fig_labels_fast_NN = ['cm_fast_NN.pdf',
                     'cm_PCR_fast_NN.pdf',
                     'f_fig_fast_NN.pdf',
                     's_fig_fast_NN.pdf']

fig_labels_fast_KNN = ['cm_fast_KNN.pdf',
                     'cm_PCR_fast_KNN.pdf',
                     'f_fig_fast_KNN.pdf',
                     's_fig_fast_KNN.pdf']



solve_slow_KNN(k_neighbors=1,
                folder_path = folderPath,
                figure_names=fig_labels_slow_NN,
                txt_filname=txt_filename_1,
                save_figures=True,
                 save_stats=True )

solve_slow_KNN(k_neighbors=7,
                folder_path = folderPath,
                figure_names=fig_labels_slow_KNN,
                txt_filname=txt_filename_2,
                save_figures=True,
                save_stats=True )


solve_fast_KNN(k_neighbors=1,
                folder_path = folderPath,
                figure_names=fig_labels_fast_NN,
                txt_filname=txt_filename_3,
                save_figures=True,
                save_stats=True)

solve_fast_KNN(k_neighbors=7, 
                folder_path = folderPath,
                figure_names=fig_labels_fast_KNN,
                txt_filname=txt_filename_2,
                save_figures=True,
                save_stats=True)
