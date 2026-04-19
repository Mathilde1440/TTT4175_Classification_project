import pandas as pd

class iris_data_class:

    def __init__(self, file_path, number_of_classes, number_of_features, class_size, class_lables, column_labels):

        self.dataframe = pd.read_csv(file_path, header=None, names= column_labels)
        self.number_of_calsses = number_of_classes
        self.number_of_features = number_of_features
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
    
    def remove_data_colum(self, column_index):
        new_def = self.dataframe.drop(self.dataframe.columns[column_index], axis = 1)
        return new_def

    def generate_plotting_data(self, filnames, relative_filepath):
        class_data = {}

        for label in range(self.number_of_features):
            temp = pd.DataFrame()
            class_data[self.column_labels[label]] = temp

        for class_index in range(len(filnames)):
            full_filpath = relative_filepath + filnames[class_index]
            temp_data_frame = pd.read_csv(full_filpath, header=None, names= self.column_labels[:-1])
            class_label = self.class_lables[class_index]
            
            for feature_index in range(self.number_of_features):
                feature_label = self.column_labels[feature_index]
                class_data[self.column_labels[feature_index]][class_label] = temp_data_frame[feature_label].values

        return class_data
    
    def remove_datacolum_from_plotting_data(self, dataFrame_dict, featur_label):
        if featur_label in dataFrame_dict:
            del dataFrame_dict[featur_label]
            


data_class = iris_data_class(
    #file_path="iris_task/iris_data_sets/iris.csv",
    file_path="iris_data_sets/iris.csv",
    number_of_classes=3,
    number_of_features=4,
    class_size=50,
    class_lables=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    column_labels=['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# new_df = data_class.remove_data_colum(0)
# print(new_df)
# traning_data_set = data_class.generate_dataset(0,30)
# plotting_set = data_class.generate_plotting_data(['class_1_csv.csv','class_2_csv.csv','class_3_csv.csv'],'iris_task/iris_data_sets/')
# print(plotting_set)
# data_class.remove_datacolum_from_plotting_data(plotting_set, 'sepal_lenght')
# print(plotting_set)





