import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model.h5")

def prepare_image(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (28,28) -> (28,28,1)
    img_array = np.expand_dims(img_array, axis=0)  # (1,28,28,1)
    return img_array

if __name__ == "__main__":
    image_path = "test_letter.png"
    img = prepare_image(image_path)
    print(img)


    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title("Test image")
    plt.axis('off')
    plt.show()

    prediction = model.predict(img)
    print(prediction)
    predicted_class = np.argmax(prediction[0])
    print(predicted_class)

    print("\nNumber certainty:")
    for i, prob in enumerate(prediction[0]):
        print(f"{i}: {prob:.3f}")
