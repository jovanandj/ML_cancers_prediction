import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

file_input = "D1550"

# Numpy load file
ml_input_scaled = np.load(file_input + "_traditional_SVD_300_scaled.npy")
labels = pd.read_csv(file_input + "_labels.csv")

# y = labels['labels'].to_list()
y = labels['labels'].values.tolist()

X = ml_input_scaled
print("Data separation done! Starting training...")

# Lists for saving the results
rf_accuracy_std = []
rf_auc_std = []
rf_sensitivity_std = []
rf_specificity_std = []

dt_accuracy_std = []
dt_auc_std = []
dt_sensitivity_std = []
dt_specificity_std = []

knn_accuracy_std = []
knn_auc_std = []
knn_sensitivity_std = []
knn_specificity_std = []

mlp_accuracy_std = []
mlp_auc_std = []
mlp_sensitivity_std = []
mlp_specificity_std = []

# TRADITIONAL ML ALGORITHMS: random forest, decision tree, K-nearest neighbour and multilayer perceptron.
names = ["Random Forest", "Decision tree", "KNN", "MLP"]

classifiers = [
    RandomForestClassifier(max_depth=10, n_estimators=100),
    DecisionTreeClassifier(random_state=0),
    KNeighborsClassifier(n_neighbors=3),
    MLPClassifier()
]
# Repeat 5 times training and testing procedure for each model
for iteration in range(5):
    # Preparing data for training and testing
    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Iteration:", iteration)
    print("RF")
    print(rf_accuracy_std)
    print(rf_auc_std)
    print(rf_sensitivity_std)
    print(rf_specificity_std)
    print("\n")

    print("DT")
    print(dt_accuracy_std)
    print(dt_auc_std)
    print(dt_sensitivity_std)
    print(dt_specificity_std)
    print("\n")

    print("KNN")
    print(knn_accuracy_std)
    print(knn_auc_std)
    print(knn_sensitivity_std)
    print(knn_specificity_std)
    print("\n")

    print("MLP")
    print(mlp_accuracy_std)
    print(mlp_auc_std)
    print(mlp_sensitivity_std)
    print(mlp_specificity_std)

    for name, clf in zip(names, classifiers):

        # Train and evaluate the model on the test set.
        # The models were evalueated using: accuracy, AUROC score, sensitivity and specificity.

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        predictions = clf.predict(X_test)
        prediction_proba = clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test, prediction_proba[:, 1].round())
        recall_sensitivity = metrics.recall_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

        if (name == "Random Forest"):
            rf_accuracy_std.append(round(score, 4))
            rf_auc_std.append(round(auc, 4))
            rf_sensitivity_std.append(round(recall_sensitivity, 4))
            rf_specificity_std.append(round(tn / (tn + fp), 4))
            print("RF done!")
        elif (name == "Decision tree"):
            dt_accuracy_std.append(round(score, 4))
            dt_auc_std.append(round(auc, 4))
            dt_sensitivity_std.append(round(recall_sensitivity, 4))
            dt_specificity_std.append(round(tn / (tn + fp), 4))
            print("DT done!")
        elif (name == "KNN"):
            knn_accuracy_std.append(round(score, 4))
            knn_auc_std.append(round(auc, 4))
            knn_sensitivity_std.append(round(recall_sensitivity, 4))
            knn_specificity_std.append(round(tn / (tn + fp), 4))
            print("KNN done!")
        else:
            mlp_accuracy_std.append(round(score, 4))
            mlp_auc_std.append(round(auc, 4))
            mlp_sensitivity_std.append(round(recall_sensitivity, 4))
            mlp_specificity_std.append(round(tn / (tn + fp), 4))
            print("MLP done!")


# Print final results for all four models.
print("RESULTS for " + file_input + "\n")

print("RF accuracy: " + str(np.mean(rf_accuracy_std)) + ", " + str(np.std(rf_accuracy_std)))
print("RF AUC: " + str(np.mean(rf_auc_std)) + ", " + str(np.std(rf_auc_std)))
print("RF Sensitivity: " + str(np.mean(rf_sensitivity_std)) + ", " + str(np.std(rf_sensitivity_std)))
print("RF Specificity: " + str(np.mean(rf_specificity_std)) + ", " + str(np.std(rf_specificity_std)))
print("\n")

print("DT accuracy: " + str(np.mean(dt_accuracy_std)) + ", " + str(np.std(dt_accuracy_std)))
print("DT AUC: " + str(np.mean(dt_auc_std)) + ", " + str(np.std(dt_auc_std)))
print("DT Sensitivity: " + str(np.mean(dt_sensitivity_std)) + ", " + str(np.std(dt_sensitivity_std)))
print("DT Specificity: " + str(np.mean(dt_specificity_std)) + ", " + str(np.std(dt_specificity_std)))
print("\n")

print("KNN accuracy: " + str(np.mean(knn_accuracy_std)) + ", " + str(np.std(knn_accuracy_std)))
print("KNN AUC: " + str(np.mean(knn_auc_std)) + ", " + str(np.std(knn_auc_std)))
print("KNN Sensitivity: " + str(np.mean(knn_sensitivity_std)) + ", " + str(np.std(knn_sensitivity_std)))
print("KNN Specificity: " + str(np.mean(knn_specificity_std)) + ", " + str(np.std(knn_specificity_std)))
print("\n")

print("MLP accuracy: " + str(np.mean(mlp_accuracy_std)) + ", " + str(np.std(mlp_accuracy_std)))
print("MLP AUC: " + str(np.mean(mlp_auc_std)) + ", " + str(np.std(mlp_auc_std)))
print("MLP Sensitivity: " + str(np.mean(mlp_sensitivity_std)) + ", " + str(np.std(mlp_sensitivity_std)))
print("MLP Specificity: " + str(np.mean(mlp_specificity_std)) + ", " + str(np.std(mlp_specificity_std)))
print("\n")