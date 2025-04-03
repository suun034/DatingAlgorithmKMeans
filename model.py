import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from faker import Faker

# Load dataset
file_path = "file.csv"  # Update with the correct file path
df = pd.read_csv(file_path)

# Initialize Faker for generating fake names
fake = Faker()

# Encoding mappings
encoding_maps = {
    "status": {"single": 0, "available": 1, "seeing someone": 2, "married": 3, "unknown": 4},
    "sex": {"m": 0, "f": 1},
    "orientation": {"straight": 0, "bisexual": 1, "gay": 2},
    "drinks": {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5},
    "drugs": {"never": 0, "sometimes": 1, "often": 2},
    "smokes": {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4},
}

# Select relevant features for clustering
features = ["age", "status", "orientation", "drinks", "drugs", "height", "smokes"]

# Ensure 'Name' column exists and fill missing names
df["Name"] = df.get("Name", np.nan)

# Add synthetic names based on gender
def generate_fake_name(sex):
    return fake.first_name_male() if sex == "m" else fake.first_name_female()

df["Name"] = df.apply(lambda row: generate_fake_name(row["sex"]) if pd.isna(row["Name"]) else row["Name"], axis=1)

# Encode categorical values
df.replace(encoding_maps, inplace=True)

# Handle missing values
for column in features:
    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    else:
        mean_value = df[column].mean()
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value if abs(df[column].skew()) > 1 else mean_value)
        df[column] = df[column].interpolate(method='linear', limit_direction='forward', axis=0)

# Scale numerical features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train K-Means clustering model
num_clusters = 247  # Change this based on the optimal number of clusters
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans_model.fit_predict(X_scaled)

# Save the K-Means model, scaler, and dataset
with open("kmeans_matchmaking_model.pkl", "wb") as file:
    pickle.dump((kmeans_model, scaler, df), file)

print("K-Means clustering model trained and saved as 'kmeans_matchmaking_model.pkl'")
