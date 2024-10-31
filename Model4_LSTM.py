# Step 0: Importing Packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Step 1: Load the Training Dataset
train_data = pd.read_csv("train.csv")  # Replace with your training dataset path
print("Training Data:")
print(train_data.head())

# Step 2: Data Preprocessing for Training Dataset
train_data = train_data.dropna()  # Drop rows with null values
X_train = train_data['Comment'].astype(str).tolist()  # Ensure features are strings
y_train = train_data['Label'].tolist()  # Target variable (labels)

# Step 3: Encode Labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Step 4: Split the Data into Training and Validation Sets
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Step 5: Tokenization and Padding
max_words = 10000  # Maximum number of words to consider
max_len = 100  # Maximum length of each input sequence

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_final)
X_train_seq = tokenizer.texts_to_sequences(X_train_final)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

# Step 6: Build the LSTM Model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))  # Embedding layer
model.add(LSTM(128, return_sequences=True))  # LSTM layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(LSTM(64))  # Another LSTM layer
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

# Step 7: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
model.fit(X_train_pad, y_train_final, epochs=2, batch_size=32, validation_data=(X_val_pad, y_val))

# Step 9: Load the Test Dataset
test_data = pd.read_csv("test.csv")
print("Test Data:")
print(test_data.head())

# Step 10: Data Preprocessing for Test Dataset
test_data = test_data.dropna()  # Drop rows with null values
X_test = test_data['Comment'].astype(str).tolist()  # Ensure features are strings

# Step 11: Tokenize and Pad the Test Dataset
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Step 12: Make Predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels

# Step 13: Prepare Results for CSV
results = pd.DataFrame({
    'ID': test_data['ID'],  # Assuming you want to keep the ID column
    'Predicted Label': label_encoder.inverse_transform(y_pred_classes)
})

# Step 19: Save Results to CSV
results.to_csv("CI_submission_6.csv", index=False)
print("Results saved to CI_submission_6.csv")
