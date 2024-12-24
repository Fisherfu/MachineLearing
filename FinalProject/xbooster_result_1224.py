# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:51:36 2024

@author: USER
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

class ImprovedXGBoostClassifier:
    def __init__(self, max_features=5000, file_path=r'C:\Users\USER\Desktop\FinalProject\Reviews.csv'):
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

    def balance_data(self, X, y):
        # 使用 SMOTE 平衡數據
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def train(self, X_train, y_train, X_test, y_test):
        # 超參數調整
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
        }
        grid_search = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # 預測
        y_pred = self.model.predict(X_test)
        print("Best Parameters:", grid_search.best_params_)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

if __name__ == "__main__":
    classifier = ImprovedXGBoostClassifier()
    X, y = classifier.preprocess_data()

    # 平衡數據
    X_balanced, y_balanced = classifier.balance_data(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    classifier.train(X_train, y_train, X_test, y_test)
