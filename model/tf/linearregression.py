import tensorflow as tf

with tf.name_scope("placeholders"):
    x = tf.compat.v1.placeholder(tf.float32, (N,1))
    y = tf.compat.v1.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    w = tf.Variable(tf.random_normal((1,1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
    y_pred = tf.matmul(x,w) + b
with tf.name_scope("loss"):
    l = tf.reduce_sum((y - y_pred)**2)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(.001).minimize(l)
with tf.name_scope("summaries"):
    tf.summary.scalar("loss",1)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("lr-train", tf.get_default_graph())

