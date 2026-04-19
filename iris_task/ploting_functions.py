import matplotlib.pyplot as plt
import seaborn as sns

def plot_double_confusion_matrix(train_confusion_1,test_confusion_1,train_confusion_2,test_confusion_2):
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
    plt.show()


def plot_histigram(dataFrame_dict , n_bins, color_lis):

    fig, ax = plt.subplots(2,2)
    row = 0
    for i, (feature_label, dataframe) in enumerate(dataFrame_dict.items()):
        idx = i%2
        if i >= 2:
            row =1

        for class_n, class_label in enumerate(dataframe.columns):
            ax[row,idx].hist(dataframe[class_label].values, n_bins, density=True,  histtype='bar', alpha=0.5 ,color =color_lis[class_n], label = class_label )
            ax[row,idx].set_title(f'{feature_label}')
        
        ax[row, idx].legend()

    plt.tight_layout()
    #plt.savefig("figures/histograms.pdf", bbox_inches="tight")
    plt.show()
