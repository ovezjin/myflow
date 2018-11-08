import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 

saver = tf.train.import_meta_graph('./model.ckpt.meta')
with tf.Session() as sess:
	saver.restore(sess, "./model.ckpt")
	print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))