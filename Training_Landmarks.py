# Note, before using, please download the following libraries
# !pip install mediapipe opencv-python pandas scikit-learn

# Import the necessary libraries/dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import matplotlib.pyplot as plt
import os

# Function to get random dataframes from the dataset for training
def do_partitioning():
    # Get coordinates from the csv file
    data_coordinates = pd.read_csv('LandMark_Coords.csv')
    
    # # Uncomment to see the first 5 data in csv
    # print(data_coordinates.head())
    # # Uncomment to see the last 5 data in csv
    # print(data_coordinates.tail())
    # # Uncomment to see specific coordinates with the classname 'Stop_Right_Hand'
    # print(data_coordinates[data_coordinates['class']=='Stop_Right_Hand'])
    
    # Setup the features and target variable for training
    X = data_coordinates.drop('class', axis=1) # features
    y = data_coordinates['class'] # target value
    
    # # Fill missing values with 0
    # X_filled = X.fillna(value=0)  
    
    # Drop columns with missing values
    X_cleaned = X.dropna(axis=1)

    # Split the data for training / do partitions with specified train_size and test_size    
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, train_size=0.8, test_size=0.2, random_state=50)

    return X_train, X_test, y_train, y_test

# Function responsible for training the partitioned dataframes
def do_training(X_train, y_train):    
    # Define pipelines for different machine learning algorithms
    pipelines = {
        'Logistic Regression':make_pipeline(StandardScaler(), LogisticRegression()),
        'Ridge Classifier':make_pipeline(StandardScaler(), RidgeClassifier()),
        'Random Forest':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'Gradient Boosting':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    } 
    
    # Create an empty dictionary to store trained models
    fit_models = {}
    
    # Iterate over each algorithm pipeline
    # Note, this may take some time to run
    for algo, pipeline in pipelines.items():
        # Fit the pipeline to the training data
        model = pipeline.fit(X_train, y_train)
        # Store the trained model in the dictionary with algorithm abbreviation as key
        fit_models[algo] = model
    
    # # Uncomment if you wish to see a sample prediction of the model
    # print(fit_models['rc'].predict(X_test))
    
    return fit_models

# Function to evaluation the accuracy of the model
def do_evaluation(trained_model, X_test, y_test):
    accuracy_scores = {}  # Dictionary to store accuracy scores
    
    for algo, model in trained_model.items():
        yhat = model.predict(X_test)
        accuracy = accuracy_score(y_test, yhat)
        accuracy_scores[algo] = accuracy
        print(algo, accuracy)
    
    return accuracy_scores     

# Visualize results of the accuracy scores based on their predictions
def visualize_results(accuracy_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy of Different Algorithms')
    plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1 for accuracy scores
    plt.show()

def save_model(trained_model, save_dir):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for algo, model in trained_model.items():
        # Construct the file path for saving the model
        file_path = os.path.join(save_dir, f"{algo}.pkl")
        
        # Save the model to the file
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    

# The main function
def main():
    # Get the partition for the test and train dataframes
    X_train, X_test, y_train, y_test = do_partitioning()
    
    # Train a model
    trained_model = do_training(X_train, y_train)
    
    # Evaluate the accuracy of the model
    accuracy_scores = do_evaluation(trained_model, X_test, y_test)
    
    # Visualize the prediction of the model
    visualize_results(accuracy_scores)
    
    # Save the model by serializing and deserializing using pickle module
    # Save the trained models to the 'models' directory
    save_model(trained_model, 'models')

if __name__ == "__main__":
    main()
