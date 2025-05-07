import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=10)
    return image, label

def preprocess_augmented(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=10)
    return image, label

if __name__ == "__main__":


    ds_train = ds_train.map(preprocess_augmented).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    stopFunction = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True);

    model.fit(ds_train, epochs=15, validation_data=ds_test, callbacks=[stopFunction])

    test_loss, test_acc = model.evaluate(ds_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    model.save("model.h5")