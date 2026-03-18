import cv2
import numpy as np

# Reads an image and returns a normalized grayscale image
def preprocess_image(path):
    img = cv2.imread(path) # Reads the image
    img = cv2.resize(img, (128, 128)) # Resizes the image to a fixed size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts the image to grayscale
    img = img / 255.0 # Normalizes pixel values to the range from 0 to 1
    return img # returns the normalized grayscale image

# Extracts frequency-domain features from an image using the Fast Fourier Transform
def fft_features(image):
    f = np.fft.fft2(image) # Computes the 2D FFT of the image to convert it to the frequency domain
    fshift = np.fft.fftshift(f) # Shifts the 0 frequency component to the center
    magnitude = np.log(np.abs(fshift) + 1) # Computes the magnitude and applies logarithmic scaling for better visibility
    spectrum = cv2.resize(magnitude, (32, 32)) # Downsamples the spectrum to a fixed size to reduce dimensionality
    return spectrum.flatten() # Returns the result as a 1D feature vector

# TODO: Function to load the training data
# TODO: Function to load the test data

def main():
    # TODO: Load training data
    # TODO: Split dataset
    # TODO: Scale features
    # Model
    # TODO: *YOUR MODEL GOES HERE*