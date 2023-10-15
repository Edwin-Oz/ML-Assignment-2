from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

model = load_model('hotdog_classifier.h5')


# Path to Not Hot dog folder
test_folder = '/Users/edwin/Documents/Machine Learning/ML-Assignment-2/test/test_not_hot_dog'

# path to hot dog folder
#test_folder = '/Users/edwin/Documents/Machine Learning/ML-Assignment-2/test/test_hot_dog'

# Set the desired number of images to process
num_images_to_process = 10000 # Process all images

# Initialize counters
total_images = 0
hot_dog_count = 0

# Iterate through the files in the test folder
for file_name in os.listdir(test_folder):
    if file_name.endswith('.jpg'):
        # Get the full path to the image
        img_path = os.path.join(test_folder, file_name)
        img = image.load_img(img_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)

        print(result[0][0])
        if result[0][0] > 0.5:
            print("Not Hot Dog.")
        else:
            print("Hot Dog!")
            hot_dog_count += 1
            
        # Update counters
        total_images += 1
        
        # Check if we've processed the desired number of images
        if total_images >= num_images_to_process:
            break
            
# Calculate the percentage
percentage_hot_dog = (hot_dog_count / total_images) * 100
print(f"Percentage of images that were hot dogs: {percentage_hot_dog}%")
    
    
    

