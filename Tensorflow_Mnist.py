

# Imports
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


# Load data
mnist = input_data.read_data_sets(".\MNIST_data", one_hot=True)
print('train images     :', mnist.train.images.shape,
      'labels:'           , mnist.train.labels.shape)
print('validation images:', mnist.validation.images.shape,
      ' labels:'          , mnist.validation.labels.shape)
print('test images      :', mnist.test.images.shape,
      'labels:'           , mnist.test.labels.shape)

# Parameters
learning_rate = 0.001
trainEpochs = 100
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)

# Network Parameters
num_hidden_neurons = 512
num_input = 784
num_classes = 10

# Define layer
def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

# Input layer
x = tf.placeholder("float", [None, num_input])

# Input label
y_label = tf.placeholder("float", [None, num_classes])

# Hidden layer
h1=layer(output_dim=num_hidden_neurons,input_dim=num_input, inputs=x, activation=tf.nn.relu)

# Output layer
y_predict=layer(output_dim=num_classes,input_dim=num_hidden_neurons, inputs=h1, activation=None)

# Define loss & optimizer
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)

# Evaluate model
correct_prediction = tf.equal(tf.argmax(y_label, 1),tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

loss_list=[]
accuracy_list=[]

# Start Training
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x,y_label: batch_y})
        
    loss,acc = sess.run([loss_function,accuracy],feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})

    loss_list.append(loss)
    accuracy_list.append(acc)    
    print("Train Epoch:", '%02d' % (epoch+1), "Loss=", "{:.4f}".format(loss)," Accuracy=",acc)

# Test data accuracy    
accuracy=sess.run(accuracy,feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
print("Final Accuracy:", accuracy)
