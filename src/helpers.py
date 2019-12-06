import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def cf_matrix(y_predicts, y_true, labels, model_name, save=False):
    y_pred = np.argmax(y_predicts, axis=1)
    y_true = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    min_val, max_val = 0, 15

    fig, ax = plt.subplots(figsize=(24, 24))
    ax.matshow(cm, cmap=plt.cm.Blues)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    for i in range(10):
        for j in range(10):
            c = cm[j, i]
            ax.text(i, j, str(c), va='center', ha='center')

    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.suptitle('Confusion matrix for '+model_name, size=24)
    plt.xlabel('True labeling', size=24)
    plt.ylabel('Predicted labeling', size=24)
    plt.rcParams.update({'font.size': 11})
    if save:
        plt.savefig('cmatrix_'+model_name+'.png')
