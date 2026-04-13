import iris_data_class
import ploting_functions


# ---- Lag dataset-objekt ----
iris_dataset = iris_data_class.iris_data_class(
    file_path="iris_data_sets/iris.csv",
    number_of_classes=3,
    number_of_features=4,
    class_size=50,
    class_lables=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    column_labels=['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'class']
)


# ---- Hent plotting-data ----
plot_data = iris_dataset.generate_plotting_data(
    filnames=['class_1_csv.csv','class_2_csv.csv','class_3_csv.csv'],
    relative_filepath="iris_data_sets/"
)


# ---- Plot histogrammer ----
ploting_functions.plot_histigram(
    dataFrame_dict=plot_data,
    n_bins=10,
    color_lis=['blue', 'orange', 'green']
)