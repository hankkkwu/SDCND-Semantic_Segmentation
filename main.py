#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #   Use tf.saved_model.loader.load to load the model and weights
    #   (https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/saved_model/load)
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    '''
    FCN32 uses a stride of 32 because after pool5 the spatial resolution is 2^5=32 times smaller, similarly pool4 should use a stride
    of 2^4=16 and pool3 should use a stride of 2^3=8.
    VGG16 model first uses stride 2 to upsample pool5 to the same size as pool4, then uses another stride 2 to upsample these 2 to be
    the same size as pool3, and finally upsamples these 3 all together with stride=8. So pool5 gets enlarged 2*2*8=32 times, pool4 gets
    enlarged 2*8=16 times and pool3 gets 8 times.
    The reason of doing it this way instead of using strides of 8, 16, 32 separately for each layer is to save the amount of computations.
    As in the end we are summing them together it's more efficient to sum before upsampling than after.

    for example, say image size = (224,224):
    encoder: (224,224) ->pool1-> (112,112) ->pool2-> (56,56) ->pool3-> (28,28) ->pool4-> (14,14) ->pool5-> (7,7)
    decoder: (7,7) ->upsampling1(stride=2)-> (14,14) ->upsampling2(stride=2)-> (28,28) ->final_upsampling(stride=8)-> (224,224)

    The kernel size seems more of a design choice, though it is not mentioned in the paper directly, the paper provides a link to their
            [Caffe implementation](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py#L60), which uses a
    kernel size eof 4 for stride 2 layer. Also note that the outputs of pooling layers 3 and 4 are scaled before they are fed into the
    1x1 convolutions.
    '''
    # TODO: Implement function
    # Reduce the number of filters from 4096 to 2
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1,1), name='conv_1x1')
    # Updampling with the same filters as pool4 layer
    upsampling1 = tf.layers.conv2d_transpose(conv_1x1, vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4,
                                             strides=(2,2), padding='SAME', name='upsampling_1')
    # Scale the pool4 layer
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')
    # Apply skip connection
    skip1 = tf.add(upsampling1, pool4_out_scaled, name='skip_1')

    # Updampling skip1 layer with the same filters as pool3 layer
    upsampling2 = tf.layers.conv2d_transpose(skip1, vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4,
                                             strides=(2,2), padding='SAME', name='upsampling_2')
    # Scale the pool3 layer
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    # Apply skip connection
    skip2 = tf.add(upsampling2, pool3_out_scaled, name='skip_2')
    # Final upsampling to the same size as input
    output = tf.layers.conv2d_transpose(skip2, num_classes, kernel_size=16, strides=(8,8), padding='SAME', name='output')
    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape logits and correct_label from 4D to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='fcn_logits')
    reshaped_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped_label), name='fcn_loss')

    # Add regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    # Use Adam optimizer that minimize the loss
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss, name='fcn_train_op')
    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    init = tf.global_variables_initializer()
    sess.run(init)
    print("training.....")
    print()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:batch_x, correct_label:batch_y, keep_prob:0.6, learning_rate:0.001})
            total_loss += loss
        print("EPOCH {} ...".format(epoch+1))
        print("Total loss: {:.3f}".format(total_loss))
        print()

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Placeholder
        correct_label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.double, name='learning_rate')

        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        saver = tf.train.Saver()
        epochs = 50
        batch_size = 8
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, image_input, correct_label, keep_prob, learning_rate)
        try:
            saver
        except NameError:
            saver = tf.train.Saver()
        saver.save(sess, './model.ckpt')
        print("Model saved")

        # saver.restore(sess, './model.ckpt')
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
