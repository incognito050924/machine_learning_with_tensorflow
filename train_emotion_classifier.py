import os
from models.cnn import inception_v3, simple_cnn
from utils.datasets import DatasetManager, split_data, get_nth_batch, load_npz
from utils.image_processing import resize_img_with_cv2
import tensorflow as tf
import numpy as np

# Hyper Parameters
batch_size = 32
num_epochs = 10000
input_shape = (128, 128, 1) # (299, 299, 3)
num_classes = 7
validation_split = .2
val_acc = 0
val_acc_max = 0

# Load Data
# data_loader = DatasetManager('fer2013', input_shape)
# x_data, y_data = data_loader.get_data()
# num_samples, num_classes = y_data.shape
# train_data, val_data = split_data(x_data, y_data, validation_split)
# train_faces, train_emotions = train_data
# val_faces, val_emotions = val_data
dataset_dir = os.path.join(os.getcwd(), 'ml_dataset', 'datasets')
train_faces, train_emotions, val_faces, val_emotions, test_faces, test_emotions = load_npz('fer2013', dataset_dir)

# Define Graph
# Placeholder
x = tf.placeholder("float", [None, 128, 128, 1])
y = tf.placeholder("float", [None, num_classes])
is_training = tf.placeholder(tf.bool)

# Prediction
prediction, end_points = inception_v3(x, num_classes=num_classes, is_training=is_training)
# prediction = simple_cnn(x, num_classes=num_classes, is_training=is_training)
# Loss & Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializer
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # Confirm Trainable Variables
    # trainable_weights = tf.trainable_variables()
    # for i, weight_name in enumerate(trainable_weights):
    #     wval = sess.run(weight_name)
    #     print("[%d/%d]\tShape: %s\t[%s]" % (i, len(trainable_weights), wval.shape, weight_name))

    # Saver
    savedir = 'nets/slim_inception_v3/'
    saver = tf.train.Saver(max_to_keep=100)
    save_step = 1
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    best_epoch = 0

    # Train model(Optimize)
    for epoch in range(num_epochs):
        avg_cost = 0.
        total_batch, remains = divmod(train_emotions.shape[0], batch_size)
        if remains > 0:
            total_batch += 1

        # Iteration
        for i in range(total_batch):
            batch_xs, batch_ys = get_nth_batch(train_faces, train_emotions, i, batch_size=batch_size)
            feed_dict = {x: batch_xs, y: batch_ys, is_training: True}
            optm, loss, acc = sess.run((optimizer, cost, accuracy), feed_dict=feed_dict)
            #avg_cost += sess.run(cost, feed_dict=feed_dict)
            avg_cost += loss
            print('Accuracy: ', acc, '\tLoss: ', loss)
        avg_cost = avg_cost / total_batch

        # Display
        print('Epoch: %03d/%03d\tAverage Loss: %.9f' % (epoch+1, num_epochs, avg_cost))
        randidx = np.random.permutation(train_faces.shape[0])[:50]
        feed_dict = {x: train_faces[randidx], y: train_emotions[randidx], is_training: False}
        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print('TRAIN ACCURACY: %.5f' % (train_accuracy))
        randidx = np.random.permutation(val_faces.shape[0])[:50]
        feed_dict = {x: val_faces[randidx], y: val_emotions[randidx], is_training: False}
        validation_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print('VALIDATION ACCURACY: %.5f' % (validation_accuracy))

        # Save
        if (epoch+1) % save_step == 0:
            savename = savedir + 'net-' + str(epoch) + '.ckpt'
            saver.save(sess=sess, save_path=savename)
            print(' [%s] SAVED.' % (savename))

        # Maximun Validaion Accuracy
        if val_acc > val_acc_max:
            val_acc_max = val_acc
            best_epoch = epoch
        print('\x1b[31m BEST EPOCH UPDATED!! [%d] \x1b[0m' % (best_epoch))
    print("OPTIMIZATION FINISHED")

    restorename = savedir + 'net-0.ckpt'
    print('LOADING [%s]' % (restorename))
    saver.restore(sess, restorename)
    randidx = np.random.permutation(test_faces.shape[0])[:50]
    feed_dict = {x: test_faces[randidx], y: test_emotions[randidx], is_training: False}
    test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    print('TEST ACCURACY: %.5f' % (test_accuracy))
