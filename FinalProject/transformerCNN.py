import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt

# 為了可以傳入keras input layer
class BertLayer(Layer):
    def __init__(self, bert_model_name='bert-base-uncased', trainable_layers=4, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(bert_model_name)

        # 將 BERT 的部分層設置為不可訓練
        total_layers = len(self.bert.bert.encoder.layer)  # BERT 的總層數
        for idx, layer in enumerate(self.bert.bert.encoder.layer):
            if idx < total_layers - trainable_layers:
                layer.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state   # 使用池化輸出
  # 使用池化輸出

class TransformerCNNClassifier:
    def __init__(self, max_len=512, num_classes=5, learning_rate=2e-5, file_path='./Reviews.csv'):
        self.max_len = max_len
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.file_path = file_path
    
    
    def preprocess_data(self):
        # 加載數據並進行預處理
        df = pd.read_csv(self.file_path)
        df = df[['Text', 'Score']].dropna().head(10000)

        # 將標籤從 1-5 映射到 0-4
        texts, labels = df['Text'], df['Score'] - 1

        # 清理文本數據
        texts = texts.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
        texts = texts.apply(lambda x: ' '.join(x.split()[:self.max_len]))  # 截斷長文本

        # 使用 Tokenizer 處理文本
        encodings = self.tokenizer(
            list(texts), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf'
        )

        return encodings['input_ids'], encodings['attention_mask'], labels.values
    def build_model(self):
        # Define Transformer + CNN model
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        bert_output = BertLayer(bert_model_name='bert-base-uncased')([input_ids, attention_mask])      

        # CNN layers with Dropout
        cnn_layer = Conv1D(filters=128, kernel_size=3, activation="relu")(bert_output)
        cnn_layer = Dropout(0.3)(cnn_layer)
        cnn_layer = GlobalMaxPooling1D()(cnn_layer)

        # Fully connected layers
        dense_layer = Dense(128, activation="relu")(cnn_layer)
        dense_layer = Dropout(0.5)(dense_layer)
        output = Dense(self.num_classes, activation="softmax")(dense_layer)

        # Compile model with AdamW
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=1e-4)
        self.model = Model(inputs=[input_ids, attention_mask], outputs=output)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=20):
        # Define callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        return history
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.2f}")
        return loss, accuracy
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions.argmax(axis=-1)  # Return the class with the highest probability

    def plot_history(self,history):
        plt.figure(figsize=(12, 5))
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    classifier = TransformerCNNClassifier()
    X, masks, y = classifier.preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X.numpy(), y, masks.numpy(), test_size=0.2, random_state=42
    )

    # Convert back to TensorFlow tensors
    X_train, X_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test)
    mask_train, mask_test = tf.convert_to_tensor(mask_train), tf.convert_to_tensor(mask_test)
    print("X_train shape:", X_train.shape)
    print("mask_train shape:", mask_train.shape)
    # Build, train, and evaluate the model
    classifier.build_model()
    history = classifier.train([X_train, mask_train], y_train, [X_test, mask_test], y_test)
    classifier.evaluate([X_test, mask_test], y_test)
    classifier.plot_history(history)

    # Save predictions
    predictions = classifier.predict([X_test, mask_test])
    pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).to_csv('evaluation_results.csv', index=False)
    classifier.model.save('transformer_cnn_model.h5')
