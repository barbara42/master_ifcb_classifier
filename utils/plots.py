# plots.py: Functions to plot training/validation accuracy over epochs, 
# and class performance metrics.

# Re-implementing the plots.py content after the reset.

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np

def plot_class_performance(precision, recall, f1, accuracy, class_names):
    """
    Plots class performance metrics including recall, precision, f1 score, and accuracy for each class.

    Args:
        precision (list): Precision values for each class.
        recall (list): Recall values for each class.
        f1 (list): F1 scores for each class.
        accuracy (list): Accuracy values for each class.
        class_names (list): Names of the classes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Class Performance')

    # Sorting classes by abundance (assuming accuracy as a proxy for abundance here for simplicity)
    indices = np.argsort(accuracy)[::-1]  # Sort in descending order

    axs[0, 0].bar(range(len(class_names)), np.array(precision)[indices])
    axs[0, 0].set_title('Precision')
    axs[0, 0].set_xticks(range(len(class_names)))
    axs[0, 0].set_xticklabels(np.array(class_names)[indices], rotation=90)

    axs[0, 1].bar(range(len(class_names)), np.array(recall)[indices])
    axs[0, 1].set_title('Recall')
    axs[0, 1].set_xticks(range(len(class_names)))
    axs[0, 1].set_xticklabels(np.array(class_names)[indices], rotation=90)

    axs[1, 0].bar(range(len(class_names)), np.array(f1)[indices])
    axs[1, 0].set_title('F1 Score')
    axs[1, 0].set_xticks(range(len(class_names)))
    axs[1, 0].set_xticklabels(np.array(class_names)[indices], rotation=90)

    axs[1, 1].bar(range(len(class_names)), np.array(accuracy)[indices])
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].set_xticks(range(len(class_names)))
    axs[1, 1].set_xticklabels(np.array(class_names)[indices], rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Creates an interactive plot of the confusion matrix using Plotly.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): Names of the classes.
    """
    # Generate the confusion matrix figure
    fig = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Viridis')
    fig.update_layout(title_text='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label'))
    fig.show()

# These functions are designed to provide visual insights into the performance of classification models,
# facilitating a deeper understanding of how well the model performs across different classes.
