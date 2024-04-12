# For localization and folder stuff
import os
# Numpy arrays needed
import numpy as np
# To load images, and convert to grayscale
from skimage import io, img_as_ubyte
# Label re-encoder
from sklearn.preprocessing import LabelEncoder
# Procedurally, randomly split data into train and test folds
from sklearn.model_selection import train_test_split
# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Descriptive statistics for results
from sklearn.metrics import classification_report

# Function to compute the histogram of an image, grayscale
def compute_histogram(image, bins=256):
    # Histogram of the flattened image
    hist, _ = np.histogram(image.ravel(), bins=bins)
    # Return a normalized histogram
    return hist / np.sum(hist)

# Parameters for the dataset 
num_samples = 40
len_feature_vector = 256
# Parameters for the machine learning algorithm
train_percent=0.2
# Provide the path to the folder containing the subfolders of images
folder_path = 'images'
# Initialize feature vectors
image_data = np.empty((num_samples,len_feature_vector), dtype=np.float64)
# Initialize label vector
labels = np.empty((num_samples), dtype=int)

sample_count = 0
# For each subfolder in "images"
for label, category in enumerate(os.listdir(folder_path)):
    # Generate the subfolder location using os.path.join for localization
    category_path = os.path.join(folder_path, category)
    # Validation: If the previous instruction was a valid path ...
    if os.path.isdir(category_path):
        # For each image in the subfolder location
        for image_file in os.listdir(category_path):
            # Again, use os.path.join to create a full path for localization
            image_path = os.path.join(category_path, image_file)
            # If the image is a JPG or other type ...
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Read the image and force it to be grayscale
                image = io.imread(image_path, as_gray=True)
                # Ensure image is uint8
                image = img_as_ubyte(image)
                # Validation: Make sure io.imread() returned something
                if image is not None:
                    # Append the histogram into the feature vector
                    image_data[sample_count,:] = compute_histogram(image)
                    # Append the label into the vector of labels
                    labels[sample_count] = label
                    # Increase the sample count
                    sample_count = sample_count + 1

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=train_percent, random_state=1234)

# Use KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the performance of the classifier
print("Classification Report:")
print(classification_report(y_test, y_pred))