from sklearn import metrics

import matplotlib.pyplot as plt

import pickle


def show_confusion_matrix(label_test, prediction):
    confusion_matrix = metrics.confusion_matrix(label_test, prediction)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["0", "1"])

    cm_display.plot()
    plt.show()


def save_model(model, history, path_name):
    model_json = model.to_json()
    with open(f"{path_name}.json","w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{path_name}.h5")
    print("Saved model to disk")
    with open(f"{path_name}.pickle", "wb") as f:
        pickle.dump(history.history, f)
