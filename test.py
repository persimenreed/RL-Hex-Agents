import os
import time
import tensorflow as tf
from multiprocessing import Process

def training():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam', loss='mse')

    x_train = tf.random.normal((1000, 32))
    y_train = tf.random.normal((1000, 10))

    print("Starting training...")
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    print("Training finished.")

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available. Running on GPU.")
    else:
        print("GPU not available. Running on CPU.")

    try:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            proc = Process(target=training)
            proc.start()

            TRAIN_SECONDS = 100
            proc.join(TRAIN_SECONDS)
            if proc.is_alive():
                print("Training time exceeded. Terminating...")
                proc.terminate()
                proc.join()
            else:
                print("Training completed within time.")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

if __name__ == "__main__":
    main()
