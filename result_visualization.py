import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List, Dict

def run_confusion_matrix(label_true, label_pred, label_name, title, save_path=None, dpi=300):
    """
    Usage: Drawing the confusion matrix of each ML model

    Args:
    label_true: True label
    label_pred: Prediction label
    label_name: Label name
    title: The title of figure
    save_path:  Saving path(xxx.png, etc.)
    dpi: File resolution(Normally 300dpi)

    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(label_true, label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("True label")
    plt.xticks(range(len(label_name)), label_name, rotation = 45)
    plt.yticks(range(len(label_name)), label_name)

    plt.tight_layout()

    # set text in confusion matrix
    for i in range(len(label_name)):
        for j in range(len(label_name)):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = format(cm[i, j], '.2f')
            plt.text(i, j, value, color = color, va='center', ha='center')
    
    if not save_path is None:
        plt.savefig(save_path, bbox_inches = 'tight', dpi = dpi)

def run_plot_metrics(metrics: Dict[str, List], save_path: str) -> None:
    """
    Usage: Do the comparison between different models

    Args: metrics (Dict[str, List]): Dictionary containing model metrics.

    """ 
    models = metrics['model']
    accuracy = metrics['accuracy']
    recall = metrics['recall']
    precision = metrics['precision']

    x = np.arange(len(models))
    width = 0.25

    # bar plot
    fig, ax = plt.subplots(figsize=(12,6))
    bar1 = ax.bar(x-width, accuracy, width, label = 'Accuracy')
    bar2 = ax.bar(x, recall, width, label = 'Recall')
    bar3 = ax.bar(x+width, precision, width, label = 'Precision')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Models by Accuracy, Recall and Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    plt.legend()

    def add_labels(figs):
        for fig in figs:
            height = fig.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(fig.get_x() + fig.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)

    fig.tight_layout()
    if not save_path is None:
        plt.savefig(save_path, dpi=300)
    # plt.show()

def run_plot_topwords(model, n_topword, features_name, title, save_path):
    """
    Usage: plot topwords

    Args: 
    model: Topic Modeling model
    n_topword: number of topwords
    features_name(list): The name of feature
    title: figure title
    save_path: saving path

    """ 
    fig, axes = plt.subplots(2, 3, figsize=(14,7), sharex=True)
    axes = axes.flatten()
    for topic_id, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_topword:]
        top_features = features_name[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_id]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_id + 1}")

        # Hide the border of the bar graph to make the image clearer.
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    if not save_path is None:
        plt.savefig(save_path, dpi=300)
    #  plt.show()

