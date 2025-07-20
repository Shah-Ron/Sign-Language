import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from collections import Counter

LABELS_CSV = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\raw\labels.csv"
NPPATH = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\processed"
SEQ_LEN = 30  # number of frames to keep per video

# Load labels
df = pd.read_csv(LABELS_CSV)
videoid_to_gloss = dict(zip(df["video_id"].astype(str), df["gloss"]))

# Step 1: Collect raw labels and feature paths
samples = []

for file in tqdm(os.listdir(NPPATH)):
    if not file.endswith(".npy"):
        continue

    video_id = os.path.splitext(file)[0]
    label = videoid_to_gloss.get(video_id)
    if label is None:
        continue

    samples.append((os.path.join(NPPATH, file), label))

# Step 2: Filter out rare labels
label_counts = Counter([label for _, label in samples])
samples = [(path, label) for path, label in samples if label_counts[label] >= 2]

# Step 3: Prepare X and y
X, y = [], []

for path, label in samples:
    data = np.load(path)

    # Padding or truncating
    if len(data) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(data), data.shape[1]))
        data = np.concatenate([data, pad])
    else:
        data = data[:SEQ_LEN]

    X.append(data)
    y.append(label)

# Step 4: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

X = np.array(X)
y = np.array(y_onehot)

# Save all data if needed
np.save("X.npy", X)
np.save("y.npy", y)
np.save("label_classes.npy", le.classes_)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=np.argmax(y, axis=1),
    test_size=0.3,
    random_state=42
)

# Step 6: Save splits
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Dataset ready!")
print(f"Total samples: {len(X)}")
print(f"Unique classes: {len(le.classes_)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
