import joblib

# Load the saved preprocessor (should be a ColumnTransformer with both scaling and encoding)
preprocessor = joblib.load('preprocessor.pkl')

# Function to preprocess user input
def preprocess_data(input_data):
    # Apply the same transformation pipeline as during training
    input_data_transformed = preprocessor.transform(input_data)
    return input_data_transformed
