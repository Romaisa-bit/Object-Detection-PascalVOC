import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 20

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Add your dataset loading code here

model.save("object_detection_model.keras")
print("Model saved successfully")
