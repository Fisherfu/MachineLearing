import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Bidirectional, LSTM, Dense, Dropout, Attention
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import transformerCNN as tc 
# 繼承 BertLayer
class BertLayer(Layer):
    def __init__(self, bert_model_name='bert-base-uncased', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(bert_model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # 返回最後一層的隱藏狀態

class TransformerBiLSTMAttentionClassifier(tc.TransformerCNNClassifier):
    def build_model(self):
        # 定義輸入
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        # BERT 層
        bert_outputs = BertLayer(bert_model_name='bert-base-uncased')([input_ids, attention_mask])

        # BiLSTM 層
        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bert_outputs)

        # 注意力層
        attention_output = Attention()([bilstm_layer, bilstm_layer])

        # 池化層
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        # 全連接層
        x = Dense(128, activation="relu")(pooled_output)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes+1, activation="softmax")(x)

        # 定義模型
        self.model = Model(inputs=[input_ids, attention_mask], outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

if __name__ == "__main__":
    classifier = TransformerBiLSTMAttentionClassifier()
    X, masks, y = classifier.preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X.numpy(), y, masks.numpy(), test_size=0.2, random_state=42
    )

    # Convert back to TensorFlow tensors
    X_train, X_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test)
    mask_train, mask_test = tf.convert_to_tensor(mask_train), tf.convert_to_tensor(mask_test)

    # Build, train, and evaluate the model
    classifier.build_model()
    history = classifier.train([X_train, mask_train], y_train, [X_test, mask_test], y_test)
    classifier.evaluate([X_test, mask_test], y_test)

    # Plot training history
    classifier.plot_history(history)

    # Save predictions
    predictions = classifier.predict([X_test, mask_test])
    pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).to_csv('bilstm_attention_results.csv', index=False)
    classifier.model.save('bilstm_attention_model.h5')
