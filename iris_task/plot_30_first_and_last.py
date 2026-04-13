import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from iris_training_model import LCD_training_model, class_1, class_2, class_3, class_labels, dataframe_to_dataset


# -------- Scenario 1: First 30 for training, last 20 for testing --------
train_df_1 = pd.concat([class_1.iloc[0:30], class_2.iloc[0:30], class_3.iloc[0:30]])
test_df_1 = pd.concat([class_1.iloc[30:50], class_2.iloc[30:50], class_3.iloc[30:50]])

training_set_1 = dataframe_to_dataset(train_df_1)
testing_set_1 = dataframe_to_dataset(test_df_1)

model_1 = LCD_training_model(
    training_set=training_set_1,
    testing_set=testing_set_1,
    number_of_classes=3,
    number_of_input_variables=4,
    class_labels=class_labels,
    alpha=0.01,
    iterations=3000,
    random_initial_weight_matrix=True
)

model_1.train()

_, _, _, train_confusion_1 = model_1.test(training_set_1)
_, _, _, test_confusion_1 = model_1.test(testing_set_1)


# -------- Scenario 2: Last 30 for training, first 20 for testing --------
train_df_2 = pd.concat([class_1.iloc[20:50], class_2.iloc[20:50], class_3.iloc[20:50]])
test_df_2 = pd.concat([class_1.iloc[0:20], class_2.iloc[0:20], class_3.iloc[0:20]])

training_set_2 = dataframe_to_dataset(train_df_2)
testing_set_2 = dataframe_to_dataset(test_df_2)

model_2 = LCD_training_model(
    training_set=training_set_2,
    testing_set=testing_set_2,
    number_of_classes=3,
    number_of_input_variables=4,
    class_labels=class_labels,
    alpha=0.01,
    iterations=3000,
    random_initial_weight_matrix=True
)

model_2.train()

_, _, _, train_confusion_2 = model_2.test(training_set_2)
_, _, _, test_confusion_2 = model_2.test(testing_set_2)


# -------- Plotting --------
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle("Confusion Matrices for Both Training Scenarios", fontsize=12)

labels = ["Setosa", "Versicolor", "Virginica"]

sns.heatmap(train_confusion_1, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[0, 0])
axes[0, 0].set_title("First 30 for training")

sns.heatmap(test_confusion_1, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[0, 1])
axes[0, 1].set_title("Last 20 for testing")

sns.heatmap(train_confusion_2, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[1, 0])
axes[1, 0].set_title("Last 30 for training")

sns.heatmap(test_confusion_2, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[1, 1])
axes[1, 1].set_title("First 20 for testing")

for ax in axes[0]:
    ax.set_xlabel("")
for ax in axes[1]:
    ax.set_xlabel("Predicted Label")

for ax in axes[:, 0]:
    ax.set_ylabel("True Label")
for ax in axes[:, 1]:
    ax.set_ylabel("")

plt.tight_layout()
#plt.savefig("figures/first_and_last_30_for_training_plot.pdf", bbox_inches="tight")
plt.show()

# -------- MSE plotting --------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle("Training Convergence (MSE)", fontsize=12)

axes[0].plot(model_1.MSE_vector)
axes[0].set_title("First 30 for training")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")
axes[0].grid(True, alpha=0.3)

axes[1].plot(model_2.MSE_vector)
axes[1].set_title("Last 30 for training")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig("figures/MSE_during_training_plot.pdf", bbox_inches="tight")
plt.show()