import transformerCNN as tc 
import biLstmAttention as bla
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import xBooster as xb
# Example usage
file_path = './Reviews.csv'


# Load dataset
df = pd.read_csv(file_path)
df = df[['Text', 'Score']].head(10000)

# Keep the original 1-5 scores for multi-class classification
texts, labels = df['Text'], df['Score']



# Initialize and preprocess data
def transformerAddCNN():
    classifier = tc.TransformerCNNClassifier()
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


def biLSTMAddAttention():
    classifier = bla.TransformerBiLSTMAttentionClassifier()
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


def XBoosterClassfiy():
    classifier = xb.XGBoostClassifier()
    X, y = classifier.preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    classifier.train(X_train, y_train, X_test, y_test)


if __name__=="__main__":

    while True: 
        modelNumber = str(input("Please input your modelType\n \
        1.TransFormer+CNN 2. biLSTM+Attension 3. XBooster 4.All\n"))
        assert all([str(string).isdigit() for string in modelNumber]),"Please input number"
        modelNumber = int(modelNumber)
        if modelNumber == 1:
            transformerAddCNN()
        elif modelNumber == 2: 
            biLSTMAddAttention()
        elif modelNumber == 3 : 
            XBoosterClassfiy()
        else: 
            transformerAddCNN()
            biLSTMAddAttention()
            XBoosterClassfiy()
            
