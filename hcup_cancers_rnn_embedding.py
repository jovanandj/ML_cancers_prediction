import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Reshape, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

file_input = "D162"

rnn_input = np.load(file_input + "_embd_input.npy")
labels = pd.read_csv(file_input + "_labels.csv")

# y = labels['labels'].to_list()
y = labels['labels'].values.tolist()

# Preparing data for training and testing
# 80% training, 20% testing
X = rnn_input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Lists for saving the results
algorithm_name = []
ml_accuarcy = []
ml_auc = []
ml_precision = []
ml_recall = []

# Parameter setup
num_unit = 64
embed_dim = 500  # 100, 200, 300, 400
btch_size = 64

# RNN LSTM implemetation
inputs = Input(shape=(X.shape[1], X.shape[2],))  # Input layer
embedded_inputs = Dense(units=embed_dim)(inputs)  # Embedding layer
lstm_outputs, _, _ = LSTM(num_unit, return_sequences=True, return_state=True)(embedded_inputs)  # LSTM layer
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

# Evaluate LSTM using: accuracy, AUROC score, recall and precision
lstm_prediction = model.evaluate(np.array(X_test), np.array(y_test))
print("RNN LSTM: " + str(lstm_prediction) + "\n\n")
rnn_proba = model.predict(X_test).squeeze()
rnn_auc_lstm = metrics.roc_auc_score(y_test, rnn_proba[:, 0])
lstm_recall = metrics.recall_score(y_test, rnn_proba[:, 0].round())
lstm_precision = metrics.precision_score(y_test, rnn_proba[:, 0].round())

algorithm_name.append("LSTM " + str(embed_dim))
ml_accuarcy.append(round(lstm_prediction[1], 4))
ml_auc.append(round(rnn_auc_lstm, 4))
ml_precision.append(round(lstm_precision, 4))
ml_recall.append(round(lstm_recall, 4))

# RNN Bidirectioanl GRU implementation
inputs_gru = Input(shape=(X.shape[1], X.shape[2]))  # Input layer
embedded_inputs_gru = Dense(units=embed_dim)(inputs_gru)  # Embedding layer
gru_outputs = Bidirectional(GRU(int(num_unit / 2), return_sequences=True))(embedded_inputs_gru)  # GRU layer
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

# Evaluate GRU using: accuracy, AUROC score, recall and precision
gru_prediction = model_gru.evaluate(np.array(X_test), np.array(y_test))
print("RNN GRU: " + str(gru_prediction) + "\n\n")
rnn_proba_gru = model_gru.predict(X_test).squeeze()
rnn_auc_gru = metrics.roc_auc_score(y_test, rnn_proba_gru[:, 0])
gru_recall = metrics.recall_score(y_test, rnn_proba_gru[:, 0].round())
gru_precision = metrics.precision_score(y_test, rnn_proba_gru[:, 0].round())

algorithm_name.append("GRU " + str(embed_dim))
ml_accuarcy.append(round(gru_prediction[1], 4))
ml_auc.append(round(rnn_auc_gru, 4))
ml_precision.append(round(gru_precision, 4))
ml_recall.append(round(gru_recall, 4))

# Save results in the csv file
results = pd.DataFrame(list(zip(algorithm_name, ml_accuarcy, ml_auc, ml_precision, ml_recall)),
                       columns=["algorithm", 'accuracy', 'auc', "precision", "recall"])
results.to_csv(file_input + "_embd500_rnn_results.csv")
