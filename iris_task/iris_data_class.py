import pandas as pd


class iris_data_class:

    def __init__(self, file_path, number_of_classes, class_size, class_lables, column_labels):

        self.dataframe = pd.read_csv(file_path, header=None, names= column_labels)
        self.number_of_calsses = number_of_classes
        self.class_size = class_size
        self.class_lables = class_lables
        self.column_labels = column_labels
        self.classes = []
        for i in range(self.number_of_calsses):
            start_index = i*self.class_size
            (self.classes).append(self.dataframe.iloc[start_index:(start_index + self.class_size)])

    def generate_dataset(self, start_index, end_index):
        set_dataframe = pd.concat([c.iloc[start_index:end_index] for c in self.classes])
        data_k = set_dataframe[set_dataframe.columns[:-1]].values
        classes =  set_dataframe['class'].values 
        return list(zip(data_k, classes))
    
    def remove_data_colum(self, colum_index):
        self.dataframe.drop(colum_index, inplace=True)


# data_class = iris_data_class(
#     file_path="iris_task/iris_data_sets/iris.csv",
#     number_of_classes=3,
#     class_size=50,
#     class_lables=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
#     column_labels=['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'class'])


# traning_data_set = data_class.generate_dataset(0,30)
# print(traning_data_set)





