import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib  # Use joblib for better compatibility
# import pickle  # If you prefer pickle, you can use it instead

# 1. Load your data
# Replace 'your_training_data.csv' with your actual data file
data = pd.read_csv('Data/your_training_data.csv')

# 2. Prepare features and target
# Replace these column names with the actual ones in your data
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']  # Replace 'label' with your target column name

# 3. Train the model
model = RandomForestClassifier()
model.fit(X, y)

# 4. Save the model
joblib.dump(model, 'models/RandomForest.pkl')
# Or, with pickle:
# import pickle
# with open('models/RandomForest.pkl', 'wb') as f:
#     pickle.dump(model, f)

print("Model retrained and saved as models/RandomForest.pkl")