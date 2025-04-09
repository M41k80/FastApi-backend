import tensorflow as tf

# cargar el modelo Keras
model = tf.keras.models.load_model('ml_model/modelo_ventas_rossmann.keras')

# convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# guardar el modelo TFLite
with open('ml_model/modelo_ventas_rossmann.tflite', 'wb') as f:
    f.write(tflite_model)
