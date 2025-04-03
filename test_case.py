import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Load the trained K-Means model, scaler, and dataset
with open("kmeans_matchmaking_model.pkl", "rb") as file:
    kmeans_model, scaler, df = pickle.load(file)

# Encoding mappings (for decoding categorical values)
encoding_maps = {
    "status": {0: "single", 1: "available", 2: "seeing someone", 3: "married", 4: "unknown"},
    "sex": {0: "male", 1: "female"},
    "orientation": {0: "straight", 1: "bisexual", 2: "gay"},
    "drinks": {0: "not at all", 1: "rarely", 2: "socially", 3: "often", 4: "very often", 5: "desperately"},
    "drugs": {0: "never", 1: "sometimes", 2: "often"},
    "smokes": {0: "no", 1: "sometimes", 2: "when drinking", 3: "yes", 4: "trying to quit"},
}

# Function to find top matches
def find_matches(test_user):
    user_orientation = test_user["orientation"]
    user_sex = test_user["sex"]
    
    def is_match(candidate):
        candidate_sex = candidate["sex"]
        candidate_orientation = candidate["orientation"]
        if user_orientation == 0:
            return (user_sex == 0 and candidate_sex == 1 and candidate_orientation in [0, 1]) or \
                   (user_sex == 1 and candidate_sex == 0 and candidate_orientation in [0, 1])
        elif user_orientation == 1:
            return candidate_orientation in [0, 1]
        elif user_orientation == 2:
            return user_sex == candidate_sex and candidate_orientation == 2
        return False

    # Predict cluster for the test user
    test_features = np.array([[test_user["age"], test_user["status"], test_user["orientation"], 
                               test_user["drinks"], test_user["drugs"], test_user["height"], test_user["smokes"]]])
    test_features_scaled = scaler.transform(test_features)
    cluster = kmeans_model.predict(test_features_scaled)[0]

    # Find candidates in the same cluster
    cluster_candidates = df[df["Cluster"] == cluster].copy()

    # Apply matching rules
    candidates = cluster_candidates[cluster_candidates.apply(is_match, axis=1)].copy()
    
    # Compute distance-based scores
    candidate_features = scaler.transform(candidates[["age", "status", "orientation", "drinks", "drugs", "height", "smokes"]])
    match_distances = euclidean_distances(candidate_features, test_features_scaled).flatten()
    
    candidates.loc[:, "distance_score"] = match_distances
    candidates = candidates.sort_values(by=["distance_score"], ascending=True)
    
    top_matches = candidates.head(10).copy()
    
    for col, mapping in encoding_maps.items():
        if col in top_matches.columns:
            top_matches[col] = top_matches[col].map(mapping)
    
    return top_matches[["Name", "age", "status", "sex", "orientation", "distance_score"]].to_dict(orient="records")

# Test case data for a user
test_case = {
    "age": 28,
    "status": 3,  # married
    "sex": 0,  # male
    "orientation": 2,  # straight
    "drinks": 2,  # socially
    "drugs": 0,  # never
    "height": 70.0,
    "smokes": 0  # no
}

# Get matches for the test case
matches = find_matches(test_case)

# Output the result
for match in matches:
    print(match)
