from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

dense = None
weights = None
model = None
interpreter = None
input_index = None
output_index = None

class_names = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

BUCKET_NAME = "devreev-tf-models" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global weights
    global model
    global dense
    if weights is None:
        download_blob(
            BUCKET_NAME,
            "models/pretrained_model.h5",
            "/tmp/pretrained_model.h5",
        )
        # model = tf.keras.models.load_model("/tmp/pretrained_model.h5")
        # keras.applications.densenet.DenseNet121(weights='./nih/densenet.hdf5', include_top=False)
    
    if dense is None:
        download_blob(
            BUCKET_NAME,
            "models/densenetXray.hdf5",
            "/tmp/densenetXray.hdf5",
        )
        base_model = keras.applications.densenet.DenseNet121(weights='/tmp/densenetXray.hdf5', include_top=False)
        print(base_model.summary())
        x = base_model.output

        # add a global spatial average pooling layer
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(len(class_names), activation="sigmoid")(x)
    
    if model in None:

        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.load_weights("/tmp/pretrained_model.h5")


    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((320, 320)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range
    print(model.summary())
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}

