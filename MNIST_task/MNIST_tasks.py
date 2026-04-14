from number_classefier import MNIST_Classefier
import matplotlib.pyplot as plt

def solve_task_one(figure_names=None, save_figures=False):

    mnist_klassefier =  MNIST_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(slow = True,print_progress_updates=True)

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels,1)
    failure_fig = mnist_klassefier.plot_images(failed_predictions[:4], "Failed predictions",2)
    succsess_fig = mnist_klassefier.plot_images(successfull_predictions[:4], "Successful predictions",3 )

    plt.show()

    
    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        failure_fig.savefig(figure_names[1])
        succsess_fig.savefig(figure_names[2])

def solve_task_two(figure_names=None, save_figures=False, k_neighbors = 1):

    mnist_klassefier =  MNIST_Classefier(file_path ='MNIST_task/NMIST_data_sets/data_all.mat', 
                                                     num_classes=10, 
                                                     class_labels=[0,1,2,3,4,5,6,7,8,9], 
                                                     n_clusters=64, 
                                                     chunk_size=10000)
    
    error_rate, failed_predictions, successfull_predictions = mnist_klassefier.run_KNN(k_neighbors = k_neighbors , slow = False, print_progress_updates=True)

    confusion_matrix_fig = mnist_klassefier.plot_confusion_matrix("Confusion matrix", mnist_klassefier.class_labels)
    failure_fig = mnist_klassefier.plot_images(failed_predictions[:4], "Failed predictions",2)
    succsess_fig = mnist_klassefier.plot_images(successfull_predictions[:4], "Successful predictions",3 )

    plt.show()

    if(save_figures):
        confusion_matrix_fig.savefig(figure_names[0])
        failure_fig.savefig(figure_names[1])
        succsess_fig.savefig(figure_names[2])



# solve_task_one()
solve_task_two()
# solve_task_two(k_neighbors=7)
