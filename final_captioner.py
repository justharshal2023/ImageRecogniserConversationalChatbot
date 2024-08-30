from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from transformers import pipeline
import gdown
import os

git_pipe = pipeline("image-to-text", model="microsoft/git-large-textcaps")

flower_output = "Flower_classifier.h5"
flower_model_id = "1AlBunIPDg4HYYCqhcHtOiXxnPFhmsoSn"
flower_url = f"https://drive.google.com/uc?id={flower_model_id}"
if not os.path.exists(flower_output):
    gdown.download(flower_url, flower_output, quiet=False)
flower_model = load_model(flower_output)
flower_model.summary()


bird_output = "Bird_classifier.h5"
bird_model_id = "1a6vqFERbrr_Cw-NyBqVHG7fsjU2-xKJ4"
bird_url = f"https://drive.google.com/uc?id={bird_model_id}"
if not os.path.exists(bird_output):
    gdown.download(bird_url, bird_output, quiet=False)
bird_model = load_model(bird_output)
bird_model.summary()


dog_output = "DogClassifier.h5"
dog_model_id = "1UFn1NGVtP5rhvcWnAANQ_4E9YRJvDEad"
dog_url = f"https://drive.google.com/uc?id={dog_model_id}"
if not os.path.exists(dog_output):
    gdown.download(dog_url, dog_output, quiet=False)
dog_model = load_model(dog_output)
dog_model.summary()


landmark_output = "LandmarkClassifierV5.h5"
landmark_model_id = "1PXixJsrUaVcHEEC-jDlv4tHT2qrCrf5c"  # Replace with your file ID
landmark_url = f"https://drive.google.com/uc?id={landmark_model_id}"
if not os.path.exists(landmark_output):
    gdown.download(landmark_url, landmark_output, quiet=False)
landmark_model = load_model(landmark_output)
landmark_model.summary()


dog_list = [
    "Bulldog",
    "Chihuahua (dog breed)",
    "Dobermann",
    "German Shepherd",
    "Golden Retriever",
    "Husky",
    "Labrador Retriever",
    "Pomeranian dog",
    "Pug",
    "Rottweiler",
    "Street dog",
]
flower_list = [
    "Jasmine",
    "Lavender",
    "Lily",
    "Lotus",
    "Orchid",
    "Rose",
    "Sunflower",
    "Tulip",
    "daisy",
    "dandelion",
]
bird_list = [
    "Crow",
    "Eagle",
    "Flamingo",
    "Hummingbird",
    "Parrot",
    "Peacock",
    "Pigeon",
    "Sparrow",
    "Swan",
]
landmark_list = [
    "The Agra Fort",
    "Ajanta Caves",
    "Alai Darwaza",
    "Amarnath Temple",
    "The Amber Fort",
    "Basilica of Bom Jesus",
    "Brihadisvara Temple",
    "Charar-e-Sharief shrine",
    "Charminar",
    "Chhatrapati Shivaji Terminus",
    "Chota Imambara",
    "Dal Lake",
    "The Elephanta Caves",
    "Ellora Caves",
    "Fatehpur Sikri",
    "Gateway of India",
    "Ghats in Varanasi",
    "Gol Gumbaz",
    "Golden Temple",
    "Group of Monuments at Mahabalipuram",
    "Hampi",
    "Hawa Mahal",
    "Humayun's Tomb",
    "The India gate",
    "Iron Pillar",
    "Jagannath Temple, Puri",
    "Jageshwar",
    "Jama Masjid",
    "Jamali Kamali Tomb",
    "Jantar Mantar, Jaipur",
    "Jantar Mantar, New Delhi",
    "Kedarnath Temple",
    "Khajuraho Temple",
    "Konark Sun Temple",
    "Mahabodhi Temple",
    "Meenakshi Temple",
    "Nalanda mahavihara",
    "Parliament House, New Delhi",
    "Qutb Minar",
    "Qutb Minar Complex",
    "Ram Mandir",
    "Rani ki Vav",
    "Rashtrapati Bhavan",
    "The Red Fort",
    "Sanchi",
    "Supreme Court of India",
    "Swaminarayan Akshardham (Delhi)",
    "Taj Hotels",
    "The Lotus Temple",
    "The Mysore Palace",
    "The Statue of Unity",
    "The Taj Mahal",
    "Vaishno Devi Temple",
    "Venkateswara Temple, Tirumala",
    "Victoria Memorial, Kolkata",
    "Vivekananda Rock Memorial",
]


def identify_dog(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = dog_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = dog_list[predicted_class_index]

    return predicted_class_label



def identify_flower(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = flower_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = flower_list[predicted_class_index]

    return predicted_class_label



def identify_bird(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = bird_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = bird_list[predicted_class_index]

    return predicted_class_label


def identify_landmark(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get predictions
    predictions = landmark_model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    # Map the predicted class index to the class label
    predicted_class_label = landmark_list[predicted_class_index]

    return predicted_class_label


def generate_final_caption(image):
    caption_dict = git_pipe(image)
    caption = caption_dict[0]["generated_text"]
    image = image.resize((256, 256))
    caption = caption_dict[0]["generated_text"]
    phrases_to_cut = ["with the word", "that says"]
    for phrase in phrases_to_cut:
        index = caption.find(phrase)
        if index != -1:
            caption = caption[:index].strip()

    if (
        "building" in caption.lower()
        or "monument" in caption.lower()
        or "tower" in caption.lower()
    ):
        caption += "\nThe landmark is : " + identify_landmark(image)
    elif "flower" in caption.lower() or "flowers" in caption.lower():
        caption += "\nThe Flower is : " + identify_flower(image)
    elif "dog" in caption.lower() or "puppy" in caption.lower():
        caption += "\nThe Dog is : " + identify_dog(image)
    elif "birds" in caption.lower() or "bird" in caption.lower():
        caption += "\nThe Bird is : " + identify_bird(image)
    return caption
