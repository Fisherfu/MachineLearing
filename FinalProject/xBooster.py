import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

class XGBoostClassifier:
    def __init__(self, max_features=5000, file_path='./Reviews.csv'):
        self.max_features = max_features
        self.file_path = file_path
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.model = None

    def preprocess_data(self):
        # 加載數據並進行預處理
        df = pd.read_csv(self.file_path)
        df = df[['Text', 'Score']].dropna().head(10000)

        # 將標籤從 1-5 映射到 0-4
        texts, labels = df['Text'], df['Score'] - 1

        # 清理文本數據
        texts = texts.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip()

        # 使用 TF-IDF 提取特徵
        X = self.vectorizer.fit_transform(texts).toarray()
        return X, labels.values

    def train(self, X_train, y_train, X_test, y_test):
        # 使用 XGBoost 訓練
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.model.fit(X_train, y_train)

        # 預測
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

if __name__ == "__main__":
    classifier = XGBoostClassifier()
    X, y = classifier.preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    classifier.train(X_train, y_train, X_test, y_test)
