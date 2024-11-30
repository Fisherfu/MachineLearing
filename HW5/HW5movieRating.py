import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from keras import Model
from keras.layers import Input,Embedding,Flatten,Dot,Add,Dense,Dropout,Concatenate,BatchNormalization
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf 
import matplotlib.pyplot as plt

class MovieRatingRecommender():
    
    def __init__(self):
        
        # Step 1: Load the dataset
        self.file_path = r'./movieRating.csv'
        self.data = pd.read_csv(self.file_path)

    def dataPreProcessing(self):
        data = self.data
        data = data.sample(frac=1).reset_index(drop=True)
        self.n_users = data['UserID'].nunique()
        self.n_movies = data['MovieID'].nunique()
        
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        user_id_map = {id_: i for i, id_ in enumerate(data['UserID'].unique())}
        movie_id_map = {id_: i for i, id_ in enumerate(data['MovieID'].unique())}
        data['UserID'] = data['UserID'].map(user_id_map)
        data['MovieID'] = data['MovieID'].map(movie_id_map)

        train_data['UserID'] = train_data['UserID'].map(user_id_map)
        train_data['MovieID'] = train_data['MovieID'].map(movie_id_map)
        test_data['UserID'] = test_data['UserID'].map(user_id_map)
        test_data['MovieID'] = test_data['MovieID'].map(movie_id_map)
        # Input data for the model
        self.train_users = train_data['UserID'].values
        self.train_movies = train_data['MovieID'].values
        self.train_ratings = train_data['Rating'].values

        self.test_users = test_data['UserID'].values
        self.test_movies = test_data['MovieID'].values
        self.test_ratings = test_data['Rating'].values
        # Step 2: Define the Keras Model
 
    def modelBuild(self):
        embedding_size = 20
        # User embedding
        user_input = Input(shape=(1,))
        user_embedding = Embedding(input_dim=self.n_users, output_dim=embedding_size)(user_input)
        user_vector = Flatten()(user_embedding)
        # Movie embedding
        movie_input = Input(shape=(1,))
        movie_embedding = Embedding(input_dim=self.n_movies, output_dim=embedding_size)(movie_input)
        movie_vector = Flatten()(movie_embedding)
        # Dot product of user and movie embeddings
        dot_product = Dot(axes=1)([user_vector, movie_vector])

        user_bias = Embedding(input_dim=self.n_users, output_dim=1,embeddings_regularizer=tf.keras.regularizers.l2(0.01))(user_input)
        movie_bias = Embedding(input_dim=self.n_movies, output_dim=1,embeddings_regularizer=tf.keras.regularizers.l2(0.01))(movie_input)

        user_vector = Dropout(0.2)(user_vector)
        movie_vector = Dropout(0.2)(movie_vector)
        user_vector = BatchNormalization()(user_vector)
        movie_vector = BatchNormalization()(movie_vector)
        x = Concatenate()([dot_product, user_bias, movie_bias])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        rating_prediction = Add()([dot_product, user_bias, movie_bias])
        model = Model(inputs=[user_input, movie_input], outputs=rating_prediction)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
        self.model = model
    def modelFit(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            x=[self.train_users, self.train_movies],
            y=self.train_ratings,
            batch_size=64,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        self.history = history
    def modelProdict(self):
        predicted_ratings = self.model.predict([self.test_users, self.test_movies]).flatten()
        mae = mean_absolute_error(self.test_ratings, predicted_ratings)
        print("im MAE Result")
        print(mae)
    def plotCurve(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
if __name__=="__main__":
    mRRObj = MovieRatingRecommender()
    mRRObj.dataPreProcessing()
    mRRObj.modelBuild()
    mRRObj.modelFit()
    mRRObj.modelProdict()
    mRRObj.plotCurve()
    
