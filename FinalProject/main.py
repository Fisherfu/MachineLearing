import transformerCNN as tc 
import biLstmAttention as bla
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf 
import xBooster as xb
import xbooster_result_1224 as ixb
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os 

# Example usage
file_path = './Reviews.csv'


# Load dataset
df = pd.read_csv(file_path)
df = df[['Text', 'Score']].head(10000)

# Keep the original 1-5 scores for multi-class classification
texts, labels = df['Text'], df['Score']



# Initialize and preprocess data
def transformerAddCNN():
    # 載入模型
    model = load_model('transformer_cnn_model.h5', custom_objects={'BertLayer': tc.BertLayer})
    print("Model loaded successfully!")

    # 使用 TransformerCNNClassifier 進行數據預處理
    classifier = tc.TransformerCNNClassifier()
    X, masks, y = classifier.preprocess_data()

    # 分割數據集
    _, X_test, _, y_test, _, mask_test = train_test_split(
        X.numpy(), y, masks.numpy(), test_size=0.2, random_state=42
    )

    # 將測試數據轉換為 TensorFlow 張量
    X_test = tf.convert_to_tensor(X_test)
    mask_test = tf.convert_to_tensor(mask_test)

    # 模型進行預測
    predictions = model.predict([X_test, mask_test])
    predicted_classes = tf.argmax(predictions, axis=-1).numpy()  # 獲取每個樣本的預測類別

    # 計算準確率
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy: {accuracy:.2f}")

    # 計算混淆矩陣
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 打印分類報告
    class_report = classification_report(y_test, predicted_classes, digits=4)
    print("Classification Report:")
    print(class_report)

    # 保存結果到 CSV
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_classes})
    results_df.to_csv('transformer_cnn_results.csv', index=False)
    print("Predictions saved to 'transformer_cnn_results.csv'")

    # 返回結果
    return accuracy, conf_matrix, class_report

def biLSTMAddAttention():
    model = load_model('bilstm_attention_model.h5', custom_objects={'BertLayer': bla.BertLayer})
    print("Model loaded successfully!")

    # 使用預處理功能處理數據
    classifier = bla.TransformerBiLSTMAttentionClassifier()
    X, masks, y = classifier.preprocess_data()

    # 分割數據為訓練集和測試集
    _, X_test, _, y_test, _, mask_test = train_test_split(
        X.numpy(), y, masks.numpy(), test_size=0.2, random_state=42
    )

    # 將測試數據轉換為 TensorFlow 張量
    X_test = tf.convert_to_tensor(X_test)
    mask_test = tf.convert_to_tensor(mask_test)

    # 模型進行預測
    predictions = model.predict([X_test, mask_test])
    predicted_classes = tf.argmax(predictions, axis=-1).numpy()  # 獲取每個樣本的預測類別

    # 計算準確度
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy: {accuracy:.2f}")

    # 計算混淆矩陣
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 畫混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 保存實際與預測結果到 CSV 文件
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_classes})
    results_df.to_csv('bilstm_attention_results.csv', index=False)
    print("Predictions saved to 'bilstm_attention_results.csv'")

    # 返回結果 DataFrame 和準確度
    return results_df, accuracy, conf_matrix


def XBoosterClassify():
    classifier = xb.XGBoostClassifier()
    X, y = classifier.preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    classifier.train(X_train, y_train, X_test, y_test)
    
def ImproveXboosterClassify():
    classifier = ixb.ImprovedXGBoostClassifier()
    X, y = classifier.preprocess_data()

    # 平衡數據
    X_balanced, y_balanced = classifier.balance_data(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    classifier.train(X_train, y_train, X_test, y_test)


if __name__=="__main__":

    while True: 
        modelNumber = str(input("Please input your modelType\n \
        1.TransFormer+CNN 2. biLSTM+Attension 3. XBooster 4.XBoosterImprove 5.all 6. exit \n"))
        assert all([str(string).isdigit() for string in modelNumber]),"Please input number"
        modelNumber = int(modelNumber)
        if modelNumber == 1:
            transformerAddCNN()
        elif modelNumber == 2: 
            biLSTMAddAttention()
        elif modelNumber == 3 : 
            XBoosterClassify()
        elif modelNumber == 4 :
            ImproveXboosterClassify()
        elif modelNumber == 6: 
            os._exit(0)
        else: 
            transformerAddCNN()
            biLSTMAddAttention()
            XBoosterClassify()
            ImproveXboosterClassify()
            
