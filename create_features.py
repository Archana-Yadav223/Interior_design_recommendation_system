import os
import pickle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

features = []
image_paths = []

base_path = "static/images"

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    # ✅ check if folder
    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)

            try:
                img = image.load_img(img_path, target_size=(224,224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                feature = model.predict(img_array).flatten()

                features.append(feature)
                image_paths.append(img_path)

            except:
                pass

pickle.dump(features, open("features.pkl", "wb"))
pickle.dump(image_paths, open("image_paths.pkl", "wb"))
print("New features.pkl created")