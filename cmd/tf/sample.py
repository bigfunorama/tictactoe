import tensorflow as tf

def main():	
	hello = tf.constant('hello TF')
	sess = tf.Session()
	a = tf.ones((2,3))
	b = tf.ones((2,3))
	c = tf.matmul(a,tf.transpose(b))
	print(sess.run(c.eval()))

if __name__ == '__main__':
    main()
