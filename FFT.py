#Date 3/15/2026 
#Group #1
#Contributions:
#Shaima Hamdallah: Image handeling and preprocessing, FFT computation, magnitude and power spectra generation, feature extraction, and building feature vectors for ML models. 
#
#
#
#TODO: 
#Shaima: implement the feature extraction based on what features we decide to use
#build the Feature Vectors for the ML models
#Matrix and return for ML models. 
"""
Project: A Image analysis ML model using FFT, the goal of this project is 
to compare different feature extraction and modeling techniques to determin 
the best approach. 
the Forior analysis portion of this project focuses on transforming galaxy images into 
frequency domain features. 

FFT class responsibilites: 
	1. Process galaxy images (convert as needed) 
	2. compute the 2D FFT of ach of the images 
	3. Generate magnitude and power spectra 
	4. extract meaningful frequency features (have not looked into yet)
	5. build a feature vector for the ML classifiers 
	more details on the workflow and methods are provided in the class docstring below.
"""
import numpy as np
import pandas as pd
import os 
import math 
from pathlib import Path	
from PIL import Image
from sklearn.preprocessing import StandardScaler	
from sklearn.model_selection import train_test_split

class FourierTransform:
	"""
	handles the fourier feature extraction from the galaxy images
	workflow:
	1. Load and preprocess images
		a. extract the galaxy ID 
		b. convert to grayscale if needed
		c. normalize pixel values
		d. store the preprocessed images for later use
		e. handle any errors in loading or preprocessing the images, such as missing files or unsupported formats
	2. Compute 2D FFT
		a. compute the 2D FFT for each preprocessed image
		b. shift the zero frequency component to the center of the spectrum, this is using the values prvided with the images (center pixel)
		c. store the FFT results for later use in generating spectra and extracting features
		d. handle any errors in computing the FFT, such as issues with the input images or numerical instability
	3. Generate magnitude and power spectra
		a. calculate the magnitude spectrum by taking the absolute value of the FFT result
		b. calculate the power spectrum by squaring the magnitude spectrum
		c. store the magnitude and power spectra for later use in feature extraction
		d. handle any errors in generating the spectra, such as issues with the FFT results or numerical instability
	4. Extract frequency features
		a. identify the dominant frequencies in the magnitude spectrum, this could involve finding the peaks in the spectrum or using other techniques to identify significant frequency components
		b. calculate the energy distribution across different frequency bands, this could involve summing the magnitude values within specific frequency ranges or using other techniques to characterize the energy distribution
		c. store the extracted features in a structured format for later use in training ML models
	5. Build feature vectors for ML models?
	"""
	#init method to set up the class with the directory of images and initialize lists to store images and features
	#implemet galaxyID extraction (file name)
	def __init__(self, image_dir, labels_csv = None, test_image_dir = None, num_radial_bins = 12,):
			"""
			Initializes the FourierTransform class with the specified parameters.
			Parameters:
			- image_dir: The directory where the labeled galaxy images are stored.
			- labels_csv: (Optional) The path to the CSV file containing galaxy labels.
			- test_image_dir: (Optional) The directory where the test galaxy images are stored.
			- num_radial_bins: The number of radial bins to use for feature extraction (default is 12).
			"""
		
			self.image_dir = Path(image_dir) #change to the directory where the galaxy images are stored
			self.test_image_dir = Path(test_image_dir) if test_image_dir else None#change to path. 
			self.labels_csv = Path(labels_csv) if labels_csv else None
			self.num_radial_bins = num_radial_bins #number of radial bins for feature extraction, can be adjusted based on the desired level of detail in the frequency features	 
			#container for labels and target column names 
			self.labels_df = None
			self.target_columns = []
			self.images = []
			self.features = []
			#containers for when training the model, will have to look into how to split the data and what features to use for training
			self.train_ids = []
			self.train_images = []
			self.train_fft_results = []
			self.train_magnitude_spectra = []
			self.train_power_spectra = []
			self.train_features = []
			self.X_train_features = None
			self.Y_train_targets = None
			#containers for test data, will	have to look into how to handle the test data and what features to extract for testing
			self.test_ids = []
			self.test_images = []
			self.test_fft_results = []
			self.test_magnitude_spectra = []
			self.test_power_spectra = []
			self.test_features = []
			self.X_test_features = None
	#method to load the labels from the CSV file, if provided, and store the target column names for later use in training the ML model
	#add error handling for missing or malformed CSV file, and ensure that the target columns are correctly identified for training the model
	def load_labels(self):
		if self.labels_csv and self.labels_csv.exists():
			self.labels_df = pd.read_csv(self.labels_csv)
			self.target_columns = [col for col in self.labels_df.columns if col != 'GalaxyID']
			print(f"Labels loaded successfully from {self.labels_csv}. Target columns: {self.target_columns}")
		else:
			print("No valid labels CSV provided. Proceeding without labels.")
		return self.labels_df, self.target_columns

	#image preprocessing: extract the galaxyID from the file name, filename: str, galaxyID: int. 
	def extract_galaxy_id(self, filename):
		return int(Path(filename).stem) #assuming the filename format is "GalaxyID.jpg" or "GalaxyID.png", this will extract the GalaxyID as an integer for later use in matching with labels and organizing the data for training and testing the ML model.
	#here we will load the images from the specified directory, convert them to grayscale if needed, and normalize the pixel values to be in the range [0, 1]. We will also store the preprocessed images in a list for later use in computing the FFT and extracting features. We will add error handling to manage any issues that arise during loading or preprocessing, such as missing files or unsupported formats.
	#the images have to be split into training and testing sets, we will have to look into how to do this based on the dataset and the labels provided, for now we will just load all the images and preprocess them, and then we can split them later based on the GalaxyID or other criteria as needed for training the ML model.
	def load_and_preprocess_training_images(self):
		"""
			loads training images and converts them to grayscale, normalize to [0,1] and save them with the galaxy ID attached to each image
		"""
		self.train_ids = []
		self.train_images = []

		#load images from the specified directory, convert to grayscale, and normalize pixel values
		for filename in os.listdir(self.image_dir):
			#check if the file is an image (e.g., .jpg, .png) and process it, might have to change to look at the specific file types we have in the dataset
			if filename.endswith('.jpg') or filename.endswith('.png'):
				img_path = self.image_dir / filename
				galaxy_id = self.extract_galaxy_id(filename) 
				img = Image.open(img_path).convert('L')  #Convert to grayscale
				img_array = np.array(img, dtype=np.float32) / 255.0 #Normalize pixel values to [0, 1]
				self.train_ids.append(galaxy_id) #store the galaxy ID.
				self.train_images.append(img_array) #store the preprocessed image in the list of images
	def load_and_preprocess_test_images(self):
		"""
		same process as the training method here. 
		"""
		if self.test_image_dir is None:
			print("No test image directory provided. Skipping test image loading.")
			return
		self.test_ids = []
		self.test_images = []
		for filename in os.listdir(self.test_image_dir):
			if filename.endswith('.jpg') or filename.endswith('.png'):
				img_path = self.test_image_dir / filename
				galaxy_id = self.extract_galaxy_id(filename) 
				img = Image.open(img_path).convert('L')
				img_array = np.array(img, dtype=np.float32) / 255.0 #Normalize pixel values to [0, 1]
				self.test_ids.append(galaxy_id) #store the galaxy ID.
				self.test_images.append(img_array) #store the preprocessed image in the list of test images
	#compute the 2D FFT for each preprocessed image and store the results to be used later for feature extraction
	def compute_fft(self, images):
		"""
		compute the 2d fft for each of the images (list pf np.ndarray) and return a list of the fft results
		"""
		self.fft_results = []
		for img in images:
				fft_result = np.fft.fftshift(np.fft.fft2(img)) #compute the 2D FFT and shift the zero frequency component to the center of the spectrum
				self.fft_results.append(fft_result) #store the FFT result for later use in generating spectra and extracting features
		return self.fft_results

	#******still not fully sure of the 2d FFT. review later.******  
	
	def compute_fft_test(self):
		"""
		test the fft method on the test images, store aswell. 

		"""
		self.test_fft_results = self.compute_fft(self.test_images) #compute the 2D FFT for the test images and store the results for later use in generating spectra and extracting features
	#generate the magnitude and power spectra from the FFT results, which will be used for feature extraction
	def generate_spectra(self, fft_results):
		self.magnitude_spectra = []
		self.power_spectra = []
		for fft_result in fft_results:
				magnitude = np.abs(fft_result) #calculate the magnitude spectrum by taking the absolute value of the FFT result
				power = magnitude ** 2 #calculate the power spectrum by squaring the magnitude spectrum
				self.magnitude_spectra.append(magnitude)
				self.power_spectra.append(power)
		return self.magnitude_spectra, self.power_spectra
	#training spectrum on the test images, builds magnitude and power spectra for the test images. 
	def generate_spectra_test(self):
		self.test_magnitude_spectra, self.test_power_spectra = self.generate_spectra(self.test_fft_results)
	#extract the frequency features from the magnitude spectra, will have to look into what features to extract, for now we are just extracting the dominant frequency and energy distribution as examples
	def extract_features(self):
		for magnitude in self.magnitude_spectra:
				dominant_freq = np.unravel_index(np.argmax(magnitude), magnitude.shape) 
				energy_distribution = np.sum(magnitude)
				self.features.append([dominant_freq, energy_distribution])
	def build_feature_vector(self):
		"""
			builds a feature vector for each image based on the extracted frequency features.
		"""
		self.feature_vectors = []
		for feature in self.features:
			feature_vector = np.array(feature).flatten()
			self.feature_vectors.append(feature_vector)

	
