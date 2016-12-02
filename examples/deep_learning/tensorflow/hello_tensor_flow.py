#Scikit-learn + TensorFlow = Scikit Flow

# A simple Hello World! using TensorFlow
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# -> Hello, TensorFlow!
