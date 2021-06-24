import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Reshape, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

file_input = "D162"

rnn_input = np.load(file_input + "_SVD_500_input.npy")
labels = pd.read_csv(file_input + "_labels.csv")

# y = labels['labels'].to_list()
y = labels['labels'].values.tolist()
X = rnn_input

# Calculating mean and standard deviation
lstm_accuracy_std = []
lstm_auc_std = []
lstm_sensitivity_std = []
lstm_specificity_std = []

gru_accuracy_std = []
gru_auc_std = []
gru_sensitivity_std = []
gru_specificity_std = []

# Parameter setup
num_unit = 64
btch_size = 64

# Repeat 5 times training and testing procedure for both models
for iteration in range(5):
    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Iteration:", iteration)
    print("LSTM")
    print(lstm_accuracy_std)
    print(lstm_auc_std)
    print(lstm_sensitivity_std)
    print(lstm_specificity_std)
    print("\n")

    print("GRU")
    print(gru_accuracy_std)
    print(gru_auc_std)
    print(gru_sensitivity_std)
    print(gru_specificity_std)
    print("\n")

    # RNN LSTM implemetation
    inputs = Input(shape=(X.shape[1], X.shape[2],))  # Input layer
    lstm_outputs, _, _ = LSTM(num_unit, return_sequences=True, return_state=True)(inputs)  # LSTM layer
    outputs = Dense(1, activation='sigmoid')(lstm_outputs)  # Output layer (Sigmoid)
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    print('Training...')
    history = model.fit(
        np.array(X_train),
        np.array(y_train),
        batch_size=btch_size,
        epochs=20,
        validation_split=0.1,  # 10% for training validation
        callbacks=[es]  # Utilizing "Early stopping" method, during the training
    )

    model.summary()

    # Evaluate LSTM using: accuracy, AUROC score, sensitivity and specificity
    lstm_prediction = model.evaluate(np.array(X_test), np.array(y_test))
    print("RNN LSTM: " + str(lstm_prediction) + "\n\n")
    rnn_proba = model.predict(X_test).squeeze()  # Prediction
    rnn_auc_lstm = metrics.roc_auc_score(y_test, rnn_proba[:, 0])  # Calculate AUROC score
    lstm_recall = metrics.recall_score(y_test, rnn_proba[:, 0].round())  # Calculate Recall
    tn, fp, fn, tp = confusion_matrix(y_test, rnn_proba[:,
                                              0].round()).ravel()  # Find: True Negative (TN), False Positive(FP), False Negative(FN) and True Positive (TP)

    lstm_accuracy_std.append(round(lstm_prediction[1], 4))
    lstm_auc_std.append(round(rnn_auc_lstm, 4))
    lstm_sensitivity_std.append(round(lstm_recall, 4))
    lstm_specificity_std.append(round(tn / (tn + fp), 4))

    # RNN Bidirectioanl GRU implementation
    inputs_gru = Input(shape=(X.shape[1], X.shape[2]))  # Input layer
    gru_outputs = Bidirectional(GRU(int(num_unit / 2), return_sequences=True))(inputs_gru)  # GRU layer
    outputs_gru = Dense(1, activation='sigmoid')(gru_outputs)  # Output layer (Sigmoid)
    model_gru = Model(inputs_gru, outputs_gru)

    model_gru.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model_gru.summary()

    print('Training...')
    history = model_gru.fit(
        np.array(X_train),
        np.array(y_train),
        batch_size=btch_size,
        epochs=20,
        validation_split=0.1,  # 10% for training validation
        callbacks=[es]  # Utilizing "Early stopping" method, during the training
    )

    # Evaluate GRU using: accuracy, AUROC score, sensitivity and specificity
    gru_prediction = model_gru.evaluate(np.array(X_test), np.array(y_test))
    print("RNN GRU: " + str(gru_prediction) + "\n\n")
    rnn_proba_gru = model_gru.predict(X_test).squeeze()  # Prediction
    rnn_auc_gru = metrics.roc_auc_score(y_test, rnn_proba_gru[:, 0])  # Calculate AUROC score
    gru_recall = metrics.recall_score(y_test, rnn_proba_gru[:, 0].round())  # Calculate Recall
    tn, fp, fn, tp = confusion_matrix(y_test, rnn_proba_gru[:,
                                              0].round()).ravel()  # Find: True Negative (TN), False Positive(FP), False Negative(FN) and True Positive (TP)

    gru_accuracy_std.append(round(gru_prediction[1], 4))
    gru_auc_std.append(round(rnn_auc_gru, 4))
    gru_sensitivity_std.append(round(gru_recall, 4))
    gru_specificity_std.append(round(tn / (tn + fp), 4))

# PRINT RESULTS FOR BOTH MODELS
print("RESULTS for " + file_input + "\n")
print("LSTM accuracy: " + str(np.mean(lstm_accuracy_std)) + ", " + str(np.std(lstm_accuracy_std)))
print("LSTM AUC: " + str(np.mean(lstm_auc_std)) + ", " + str(np.std(lstm_auc_std)))
print("LSTM sensitivity: " + str(np.mean(lstm_sensitivity_std)) + ", " + str(np.std(lstm_sensitivity_std)))
print("LSTM specificity: " + str(np.mean(lstm_specificity_std)) + ", " + str(np.std(lstm_specificity_std)))

print("GRU accuracy: " + str(np.mean(gru_accuracy_std)) + ", " + str(np.std(gru_accuracy_std)))
print("GRU AUC: " + str(np.mean(gru_auc_std)) + ", " + str(np.std(gru_auc_std)))
print("GRU sensitivity: " + str(np.mean(gru_sensitivity_std)) + ", " + str(np.std(gru_sensitivity_std)))
print("GRU specificity: " + str(np.mean(gru_specificity_std)) + ", " + str(np.std(gru_specificity_std)))
