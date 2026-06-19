import tensorflow as tf

model = tf.keras.models.load_model(
    "object_detection_model.keras"
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

with open(
    "object_detection_model.tflite",
    "wb"
) as f:
    f.write(tflite_model)

print("TFLite model created successfully")
