import iris_data_class
import iris_training_model
import ploting_functions


iris_dataset = iris_data_class.iris_data_class(
    file_path="iris_task/iris_data_sets/iris.csv",
    number_of_classes=3,
    class_size=50,
    class_lables=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    )

training_set_first = iris_dataset.generate_dataset(0,30)
test_set_first = iris_dataset.generate_dataset(30,50)
training_set_last = iris_dataset.generate_dataset(20,50)
test_set_last = iris_dataset.generate_dataset(0,20) 

iris_model_first = iris_training_model.LCD_training_model(
    training_set=training_set_first,
    testing_set=test_set_first,
    number_of_classes=3,
    number_of_input_variables=4,
    class_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    alpha=0.01,
    iterations=3000,
    random_initial_weight_matrix=True
)

iris_model_last = iris_training_model.LCD_training_model(
    training_set=training_set_last,
    testing_set=test_set_last,
    number_of_classes=3,
    number_of_input_variables=4,
    class_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    alpha=0.01,
    iterations=3000,
    random_initial_weight_matrix=True
)


iris_model_first.train()
iris_model_last.train()

train_first_correct, train_first_error_count, train_first_error_rate, train_first_confusion_matrix = iris_model_first.test(training_set_first)
test_first_correct, test_first_error_count, test_first__error_rate, test_first_confusion_matrix = iris_model_first.test(test_set_first)

train_last_correct, train_last_error_count, train_last_error_rate, train_last_confusion_matrix = iris_model_last.test(training_set_last)
test_last_correct, test_last_error_count, test_last_error_rate, test_last_confusion_matrix = iris_model_last.test(test_set_last)

iris_model_first.print_results(train_first_correct, train_first_error_count, train_first_error_rate, train_first_confusion_matrix,test=False)
iris_model_first.print_results(test_first_correct, test_first_error_count, test_first__error_rate, test_first_confusion_matrix,test=True)
iris_model_last.print_results(train_last_correct, train_last_error_count, train_last_error_rate, train_last_confusion_matrix ,test=False)
iris_model_last.print_results(test_last_correct, test_last_error_count, test_last_error_rate, test_last_confusion_matrix ,test=True)

ploting_functions.plot_double_confusion_matrix(train_first_confusion_matrix,test_first_confusion_matrix,train_last_confusion_matrix,test_last_confusion_matrix)