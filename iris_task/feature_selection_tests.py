import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import iris_data_class
import iris_training_model


# -----------------------------
# Load dataset using your data class
# -----------------------------
iris_dataset = iris_data_class.iris_data_class(
    file_path="iris_data_sets/iris.csv",
    number_of_classes=3,
    number_of_features=4,
    class_size=50,
    class_lables=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    column_labels=['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'class']
)


# -----------------------------
# Generate train/test split using your function
# First 30 for training, last 20 for testing
# -----------------------------
training_set_full = iris_dataset.generate_dataset(0, 30)
testing_set_full = iris_dataset.generate_dataset(30, 50)


# -----------------------------
# Select only chosen features
# -----------------------------
def filter_dataset(dataset, feature_indices):
    filtered_dataset = []

    for xk, label in dataset:
        filtered_xk = xk[list(feature_indices)]

        # add bias term so the model still uses g = Wx, but with offset included
        filtered_xk = np.append(filtered_xk, 1.0)

        filtered_dataset.append((filtered_xk, label))

    return filtered_dataset


# -----------------------------
# Standardize using training-set statistics
# Do not standardize the last column, since it is the bias term
# -----------------------------
def standardize_datasets(training_set, testing_set):
    X_train = np.array([xk for xk, _ in training_set], dtype=float)
    X_test = np.array([xk for xk, _ in testing_set], dtype=float)

    X_train_features = X_train[:, :-1]
    X_test_features = X_test[:, :-1]

    mean = np.mean(X_train_features, axis=0)
    std = np.std(X_train_features, axis=0)

    std[std == 0] = 1.0

    X_train[:, :-1] = (X_train_features - mean) / std
    X_test[:, :-1] = (X_test_features - mean) / std

    training_set_std = [(X_train[i], training_set[i][1]) for i in range(len(training_set))]
    testing_set_std = [(X_test[i], testing_set[i][1]) for i in range(len(testing_set))]

    return training_set_std, testing_set_std


# -----------------------------
# Experiments after removing sepal_width
#
# Original feature order:
# 0 = sepal_lenght
# 1 = sepal_width
# 2 = petal_length
# 3 = petal_width
# -----------------------------
experiments = [
    ("Sepal Length, Petal Length, Petal Width", [0, 2, 3]),
    ("Sepal Length, Petal Length", [0, 2]),
    ("Sepal Length, Petal Width", [0, 3]),
    ("Petal Length, Petal Width", [2, 3]),
    ("Sepal Length only", [0]),
    ("Petal Length only", [2]),
    ("Petal Width only", [3]),
]


# -----------------------------
# Run experiments
# -----------------------------
results = []

for experiment_name, feature_indices in experiments:
    training_set = filter_dataset(training_set_full, feature_indices)
    testing_set = filter_dataset(testing_set_full, feature_indices)

    training_set, testing_set = standardize_datasets(training_set, testing_set)

    np.random.seed(42)

    model = iris_training_model.LCD_training_model(
        training_set=training_set,
        testing_set=testing_set,
        number_of_classes=3,
        number_of_input_variables=len(feature_indices) + 1,
        class_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        alpha=0.01,
        iterations=5000,
        random_initial_weight_matrix=True
    )

    model.train()

    train_correct, train_error_count, train_error_rate, train_confusion_matrix = model.test(training_set)
    test_correct, test_error_count, test_error_rate, test_confusion_matrix = model.test(testing_set)

    results.append({
        "name": experiment_name,
        "train_error_rate": train_error_rate,
        "test_error_rate": test_error_rate,
        "train_confusion_matrix": train_confusion_matrix,
        "test_confusion_matrix": test_confusion_matrix
    })

# -----------------------------
# Print training error rates
# -----------------------------
print("\nTraining Error Rates")
print("-" * 60)

for result in results:
    print(f"{result['name']}: {result['train_error_rate'] * 100:.2f} %")

# -----------------------------
# Print test error rates
# -----------------------------
print("\nTest Error Rates")
print("-" * 60)

for result in results:
    print(f"{result['name']}: {result['test_error_rate'] * 100:.2f} %")


# -----------------------------
# Plot all test confusion matrices
# -----------------------------
labels = ["Setosa", "Versicolor", "Virginica"]

for result in results:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(result["name"], fontsize=12)

    # Training confusion matrix
    sns.heatmap(
        result["train_confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axes[0]
    )
    axes[0].set_title("Training set")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Testing confusion matrix
    sns.heatmap(
        result["test_confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axes[1]
    )
    axes[1].set_title("Testing set")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    safe_name = result["name"].replace(",", "").replace(" ", "_").lower()

    plt.tight_layout()
    #plt.savefig(f"figures/{safe_name}.pdf", bbox_inches="tight")
    plt.show()