import cv2
import numpy as np
import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "galaxy-zoo-the-galaxy-challenge")
TARGET_COLUMNS = ["Class1.1", "Class1.2", "Class1.3", "Class2.1", "Class2.2", "Class3.1", "Class3.2", "Class4.1", "Class4.2", "Class5.1", "Class5.2", "Class5.3", "Class5.4", "Class6.1", "Class6.2", "Class7.1", "Class7.2", "Class7.3", "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6", "Class8.7", "Class9.1", "Class9.2", "Class9.3", "Class10.1", "Class10.2", "Class10.3", "Class11.1", "Class11.2", "Class11.3", "Class11.4", "Class11.5", "Class11.6"]
SAMPLE_SIZE = 3000  # Int value up to 61578 or None for all.
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def preprocess_image(raw_bytes):
    """Reads raw image bytes(from zip) and returns a normalized grayscale image"""
    decoded = np.frombuffer(raw_bytes, dtype=np.uint8) # Decodes the raw bytes into a numpy array
    img = cv2.imdecode(decoded, cv2.IMREAD_COLOR) # Decodes the array into a color image
    img = cv2.resize(img, (128, 128)) # Resizes the image to a fixed size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts the image to grayscale
    img = img / 255.0 # Normalizes pixel values to the range from 0 to 1
    return img # returns the normalized grayscale image


def extract_fft_features(image):
    """Extract FFT magnitude spectrum as a flat feature vector"""
    fft = np.fft.fft2(image) # Computes the 2D FFT of the image to convert it to the frequency domain
    fft_shift = np.fft.fftshift(fft) # Shifts the 0 frequency component to the center
    magnitude = np.log(np.abs(fft_shift) + 1) # Computes the magnitude and applies logarithmic scaling for better visibility
    spectrum = cv2.resize(magnitude, (32, 32)) # Downsamples the spectrum to a fixed size to reduce dimensionality
    return spectrum.flatten() # Returns the result as a 1D feature vector


def load_training_data(sample_size=SAMPLE_SIZE):
    """Returns (features, labels) arrays from training zips."""
    solutions_path = os.path.join(DATA_DIR, "training_solutions_rev1.zip")
    images_path = os.path.join(DATA_DIR, "images_training_rev1.zip")

    with zipfile.ZipFile(solutions_path) as zip:
        with zip.open("training_solutions_rev1.csv") as file:
            labels_df = pd.read_csv(file)

    # =min(sample_size, len(labels_df)) prevents crash if sample_size > 61578
    if sample_size is not None:
        labels_df = labels_df.sample(n=min(sample_size, len(labels_df)), random_state=RANDOM_SEED)

    galaxy_ids = labels_df["GalaxyID"].values
    labels = labels_df[TARGET_COLUMNS].values

    features = []
    valid_indices = []
    with zipfile.ZipFile(images_path) as zip:
        for i, galaxy_id in enumerate(galaxy_ids):
            try:
                with zip.open(f"images_training_rev1/{galaxy_id}.jpg") as img_file:
                    image = preprocess_image(img_file.read())
                    features.append(extract_fft_features(image))
                    valid_indices.append(i)
            except KeyError:
                continue
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(galaxy_ids)} images")

    print(f"  Loaded {len(features)} images total")
    return np.array(features), labels[valid_indices]


def load_test_data():
    """Returns (features, galaxy_ids) arrays from test zip."""
    images_path = os.path.join(DATA_DIR, "images_test_rev1.zip")

    features = []
    galaxy_ids = []
    with zipfile.ZipFile(images_path) as zip:
        filenames = [name for name in zip.namelist() if name.endswith(".jpg")]
        for i, filename in enumerate(filenames):
            # Remove .jpg so that only ID remains
            galaxy_id = int(os.path.basename(filename).replace(".jpg", ""))
            with zip.open(filename) as img_file:
                image = preprocess_image(img_file.read())
                features.append(extract_fft_features(image))
                galaxy_ids.append(galaxy_id)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(filenames)} test images")

    print(f"  Loaded {len(features)} test images total")
    return np.array(features), np.array(galaxy_ids)


def main():
    print("Loading training data")
    features, labels = load_training_data()

    print("Splitting dataset")
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    print("Training model")
    model = LinearRegression()
    model.fit(train_features, train_labels)

    predictions = model.predict(val_features)
    overall_rmse = np.sqrt(mean_squared_error(val_labels, predictions))
    print(f"Validation RMSE: {overall_rmse:.4f}")
    for i, col in enumerate(TARGET_COLUMNS):
        col_rmse = np.sqrt(mean_squared_error(val_labels[:, i], predictions[:, i]))
        print(f"  {col}: RMSE = {col_rmse:.4f}")


if __name__ == "__main__":
    main()