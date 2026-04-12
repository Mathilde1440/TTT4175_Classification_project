from iris_training_model import LCD_training_model, class_1, class_2, class_3, class_labels, dataframe_to_dataset
import pandas as pd

train_df = pd.concat([class_1.iloc[20:50], class_2.iloc[20:50], class_3.iloc[20:50]])
test_df = pd.concat([class_1.iloc[0:20], class_2.iloc[0:20], class_3.iloc[0:20]])

training_set = dataframe_to_dataset(train_df)
testing_set = dataframe_to_dataset(test_df)

model = LCD_training_model(
    training_set=training_set,
    testing_set=testing_set,
    number_of_classes=3,
    number_of_input_variables=4,
    class_labels=class_labels,
    alpha=0.01,
    iterations=3000,
    random_initial_weight_matrix=True
)

model.train()

train_correct, train_error_count, train_error_rate, train_confusion_matrix = model.test(training_set)
test_correct, test_error_count, test_error_rate, test_confusion_matrix = model.test(testing_set)

print("Training set results:")
print("Correct classifications:", train_correct)
print("Misclassifications:", train_error_count)
print("Error rate:", train_error_rate)
print("Confusion matrix:")
print(train_confusion_matrix)

print()

print("Testing set results:")
print("Correct classifications:", test_correct)
print("Misclassifications:", test_error_count)
print("Error rate:", test_error_rate)
print("Confusion matrix:")
print(test_confusion_matrix)