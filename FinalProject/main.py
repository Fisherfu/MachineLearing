import transformerCNN
import pandas as pd 
from sklearn.model_selection import train_test_split
# Example usage
file_path = '../Reviews.csv'


# Load dataset
df = pd.read_csv(file_path)
df = df[['Text', 'Score']].head(10000)

# Keep the original 1-5 scores for multi-class classification
texts, labels = df['Text'], df['Score']



# Initialize and preprocess data
def transformerAddCNN():
    classifier = transformerCNN.TransformerCNNClassifier()
    X, masks, y = classifier.preprocess_data(texts, labels)

    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X, y, masks, test_size=0.2, random_state=42
    )

    # Build, train, and evaluate the model
    classifier.build_model()
    history = classifier.train([X_train, mask_train], y_train, [X_test, mask_test], y_test)
    classifier.evaluate([X_test, mask_test], y_test)

def biLSTMAddAttention():
    pass 

def XBoosterClassfiy():
    pass 

if __name__=="__main__":

    while True: 
        modelNumber = str(input("Please input your modelType\n \
        1.TransFormer+CNN 2. biLSTM+Attension 3. XBooster 4.All"))
        assert all([str(string).isdigit() for string in modelNumber]),"Please input number"
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
            
