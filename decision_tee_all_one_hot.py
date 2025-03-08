#%% load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from get_ids import get_ids_of_easy_outliers

#load data
df = pd.read_csv("data/datathon_data.csv")


# Drop BELNR (Account Document Number) since it's just an identifier
ids_to_remove = get_ids_of_easy_outliers(df)
df = df.drop(index = ids_to_remove, columns=['BELNR'])

#%% encode
# Assuming your dataset is a pandas DataFrame called 'df'
# Select all columns that are of type 'object' (i.e., char/categorical)
categorical_columns = df.select_dtypes(include=['object']).columns

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded = df_encoded.rename(columns={"label_anomal": "label"}).drop(columns="label_regular")

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
# Define the target and features
X = df_encoded.drop(columns=['label'])  # All columns except the 'label'
y = df_encoded['label']  # The target variable

# Split the data into training and test sets
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train = X, y
X_test, y_test = X, y

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
recall = recall_score(y_test, y_pred)
print(f'Recall (True Positive Rate): {recall:.2f}')


# %%
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the decision tree
plt.figure(figsize=(20, 10))  # Adjusting figure size
plot_tree(model, 
          feature_names=X.columns,   # List of feature names
          class_names=[str(c) for c in model.classes_],  # Class names as strings
          filled=True,  # Color the nodes based on class
          fontsize=10,  # Font size for labels
          max_depth=None)  # Limit tree depth for better visualization (optional)

plt.show()

# %%
