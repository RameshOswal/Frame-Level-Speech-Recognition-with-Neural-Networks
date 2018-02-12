# Used the MNIST tutorial problem for network setup: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
FLATTENED_DATA_FOLDER_PATH = '../data/flattened'
# from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

class Data:
        def __init__(self, data_folder_path=FLATTENED_DATA_FOLDER_PATH, batch_size=1000000, num_context_frame=1):
            self.batch_size = batch_size
            self.path = data_folder_path
            self.curr_batch_idx = 0
            self.num_context_frame = num_context_frame 
        def load_train_batchwise(self):
            TRAIN = 'train'
            self.trainLookup = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, TRAIN + 'FLATTENED_Lookup.npy')).astype(int)
            self.trainXmem_map = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, TRAIN + 'FLATTENED_X.npy'),mmap_mode="r")
            self.trainYmem_map = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, TRAIN + 'FLATTENED_Y.npy'),mmap_mode="r")
            
            self.total_num_batch = int(self.trainLookup.shape[0]/self.batch_size)+1
            Lookup_indexes = np.array(range(self.trainLookup.shape[0]))
            np.random.shuffle(Lookup_indexes)
            
            for curr_batch in range(self.total_num_batch):
                print("***Current Batch:", curr_batch)
                start_idx = curr_batch * self.batch_size
                end_idx = start_idx + self.batch_size
                
                batch_indx = self.trainLookup[Lookup_indexes[start_idx: end_idx]]
                batch_matrix = np.zeros((self.num_context_frame*2+1, batch_indx.shape[0]))
                batch_matrix[self.num_context_frame] = batch_indx
                for idx in range(self.num_context_frame):
                    batch_matrix[self.num_context_frame - idx - 1] = batch_matrix[self.num_context_frame - idx ] -1
                    batch_matrix[self.num_context_frame + idx + 1 ] = batch_matrix[self.num_context_frame + idx] +1
                batch_matrix = batch_matrix.T 
                
                X, Y = self.trainXmem_map[batch_matrix.astype(int)], self.trainYmem_map[Lookup_indexes[start_idx: end_idx]]
                yield np.array(X).reshape(X.shape[0], X.shape[1]*X.shape[2]), np.array(Y)
        
        def load_dev(self):
            self.devLookup = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, 'dev' + 'FLATTENED_Lookup.npy')).astype(int)
            self.devXmem_map = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, 'dev' + 'FLATTENED_X.npy'))
            self.devYmem_map = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, 'dev' + 'FLATTENED_Y.npy'))
            
            batch_indx = self.devLookup
            batch_matrix = np.zeros((self.num_context_frame*2+1, batch_indx.shape[0]))
            batch_matrix[self.num_context_frame] = batch_indx
            
            for idx in range(self.num_context_frame):
                batch_matrix[self.num_context_frame - idx - 1] = batch_matrix[self.num_context_frame - idx ] -1
                batch_matrix[self.num_context_frame + idx + 1 ] = batch_matrix[self.num_context_frame + idx] +1
            batch_matrix = batch_matrix.T 
            
            X, Y = self.devXmem_map[batch_matrix.astype(int)], self.devYmem_map[:]
            return X.reshape(X.shape[0], X.shape[1]*X.shape[2]), Y
        def load_test(self):
            self.testLookup = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, 'test' + 'FLATTENED_Lookup.npy')).astype(int)
            self.testXmem_map = np.load(os.path.join(FLATTENED_DATA_FOLDER_PATH, 'test' + 'FLATTENED_X.npy'))
            
            batch_indx = self.testLookup
            batch_matrix = np.zeros((self.num_context_frame*2+1, batch_indx.shape[0]))
            batch_matrix[self.num_context_frame] = batch_indx
            
            for idx in range(self.num_context_frame):
                batch_matrix[self.num_context_frame - idx - 1] = batch_matrix[self.num_context_frame - idx ] -1
                batch_matrix[self.num_context_frame + idx + 1 ] = batch_matrix[self.num_context_frame + idx] +1
            batch_matrix = batch_matrix.T 
            
            X = self.testXmem_map[batch_matrix.astype(int)]
            return X.reshape(X.shape[0], X.shape[1]*X.shape[2])


# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100000
display_step = 1

# Network Parameters
n_hidden_1 = 400 # 1st layer number of neurons
n_hidden_2 = 400 # 2nd layer number of neurons
n_input = 40 # MNIST data input (img shape: 28*28)
n_classes = 138 # MNIST total classes (0-9 digits)


def one_hot(arr, num_classes=n_classes):
    return np.eye(num_classes)[arr.astype(int)]


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
   'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
SAVED_MODELS = '../saved_models/'
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        print("********************Starting Epoch: ",epoch)
        avg_cost = 0.
        d = Data(batch_size=batch_size)
        # Loop over all batches
        for batch_x, batch_y in d.load_train_batchwise():
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_y_one_hot = one_hot(batch_y, n_classes)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y_one_hot})
            # Compute average loss
            avg_cost += c / d.total_num_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    saver.save(sess, SAVED_MODELS + 'my-model')

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    devX, devY = d.load_dev() 
    print("Accuracy:", accuracy.eval({X: devX, Y: one_hot(devY, n_classes)}))
    print("**********Making Predictions:")
    print("****************Loading test data")
    testX = d.load_test()
    pred = prediction.eval({X: testX})
    final_pred = np.concatenate((np.array(range(len(pred)))[np.newaxis].T, np.array(pred)[np.newaxis].T),axis=1)
    np.savetxt(X=final_pred, fname='../data/predictions.csv',delimiter=",", header="id,label")
    print("Saved test file")
    print(final_pred)