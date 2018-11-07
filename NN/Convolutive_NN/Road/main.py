# Import module package
import os.path
import shutil
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm
import warnings
import tensorflow as tf
import numpy as np
import glob
import re
import scipy.misc
import random
import time

# **********************************************************************************************************************
# HIPER-PARAMETERS
# **********************************************************************************************************************
LEARNING_RATE = np.random.uniform(1e-5, 5 * 1e-4)
BATCH_SIZE = np.random.randint(2, 6)
KEEP_PROB = np.random.uniform(0.2, 0.7)
NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)
DATA_DIR = './data'
RUNS_DIR = './runs'
epochs = 20


""""
First, check that vgg.zip is download. If it is not downloaded, call the function maybe_download_pretrained_vgg. 
If it is downloaded, the files are extracted from get_vgg_file"
"""


# **********************************************************************************************************************
# FUNCTION: OBTAIN THE VGG FOLDER
# **********************************************************************************************************************
def find_file(format, dir_to_search):
    a = 3
    b = 4
    return a, b


# **********************************************************************************************************************
# FUNCTION: DOWNLOAD VGG.ZIP
# **********************************************************************************************************************

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

# **********************************************************************************************************************
# FUNCTION: LOAD VGG16
# **********************************************************************************************************************


def load_vgg(sess, vgg_path):
    # Load pretrained VGG Model into TensorFlow.
    # :param sess: TensorFlow Session
    # :param vgg_path: Path to vgg folder, containing "variable/ and "save_model.pb"
    # :return: Tuple of Tensors for VGG  model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

    # TODO: Implement function
        # Use tf.save.model.loader.load to load the model and weights

    # 1. Check if exists the .pb file for VGG16 in folder "VGG_model". In case it is not, then we need to transform it.
    files, nfiles = find_file(format="pb", dir_to_search=vgg_path)
    if nfiles == 0:
        warnings.warn("No pretrained VGG16 model found. Downloading pretrained VGG16.....")
        maybe_download_pretrained_vgg(vgg_path)
        print("Downloading process done!")
    else:
        print("Pretrained model already exists on dir", vgg_path)

    # 2. Define the names of the different fields to get from the model.

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_tensor_name = 'keep_prob:0'
    vgg_layer3out_tensor_name = 'layer3_out:0'
    vgg_layer4out_tensor_name = 'layer4_out:0'
    vgg_layer7out_tensor_name = 'layer7_out:0'

    # 3. Get the fields from the model

    # The tag to introduce on the <c> tf.saved_model.loader.load () function is the model 'vgg16'
    vgg_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep = tf.get_default_graph().get_tensor_by_name(vgg_keep_tensor_name)
    vgg_layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3out_tensor_name)
    vgg_layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4out_tensor_name)
    vgg_layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7out_tensor_name)

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    return vgg_input, vgg_keep, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

# **********************************************************************************************************************
# FUNCTION: CREATION LAYERS OF THE NEURAL NETWORK
# **********************************************************************************************************************


def layers(vgg_layer7, vgg_layer4, vgg_layer3, num_classes):

    # Define the Kernel initializer
    kernel_initializer = tf.random_normal_initializer(stddev=0.01)

    # Define the padding
    padding = 'SAME'

    # Define the kernel_size and stride for the convolutional layers
    conv_kernel_size = 1
    conv_stride = 1

    # Define the kernel_size and stride for the deconvolutional layers
    deconv_kernel_size = (4, 4)
    deconv_stride = (2, 2)

    # Define kernel_size and stride for the last deconvolutional layer
    deconv_last_layer_size = (16, 16)
    deconv_last_stride = (8, 8)

    # Convolutional layer number 7
    # tf.layers.conv2d(input, filters, kernel_size, strides, padding)
    conv_layer_7 = tf.layers.conv2d(vgg_layer7, num_classes, deconv_kernel_size,
                                    deconv_stride, kernel_initializer=kernel_initializer)

    # Deconvolutional layer number 1
    # tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding)
    deconv_layer_1 = tf.layers.conv2d_transpose(conv_layer_7, num_classes, deconv_kernel_size,
                                                deconv_stride, padding, kernel_initializer=kernel_initializer)

    # Convolutional layer number 4
    conv_layer_4 = tf.layers.conv2d(vgg_layer4, num_classes, conv_kernel_size,
                                    conv_stride, kernel_initializer=kernel_initializer)

    # Skip union 1
    skip_dec_1_layer_4 = tf.add(deconv_layer_1, conv_layer_4)

    # Deconvolutional layer number 2
    deconv_layer_2 = tf.layers.conv2d_transpose(skip_dec_1_layer_4, num_classes, deconv_kernel_size,
                                                deconv_stride, padding, kernel_initializer=kernel_initializer)

    # Convolutional layer number 3
    conv_layer_3 = tf.layers.conv2d(vgg_layer3, num_classes, conv_kernel_size,
                                    conv_stride, kernel_initializer=kernel_initializer)

    # Skip union 2
    skip_dec_2_layer_3 = tf.add(deconv_layer_2, conv_layer_3)

    # Deconvolutional layer 3
    out = tf.layers.conv2d_transpose(skip_dec_2_layer_3, num_classes, deconv_last_layer_size,
                                                deconv_last_stride, padding, kernel_initializer=kernel_initializer)

    return out

# **********************************************************************************************************************
# FUNCTION:  LOSS & OPTIMIZE THE NEURAL NETWORK
# **********************************************************************************************************************


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    # Build the TensorFlow loss and optimizer operations.
    # :param nn_last_layer: TF Tensor of the last layer in the neural network
    # :param correct_label: TF Placeholder for the correct label image
    # :param learning_rate: TF Placeholder for the learning rate
    # :param num_classes: Number of classes to classify
    # :return: Tuple of (logits, train_op, cross_entropy_loss)
    ## TODO: Implement function

    # Define here your train_function
    train_function = "ADAM"

    # Logits: Reshape from 4D to 2D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Labels
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # Training function definition
    if train_function == "SGD":
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
    elif train_function == "ADAM":
        train_op = tf.train.AdamOptimizer(learning_rate). minimize(cross_entropy_loss)
    else:
        print("The training function [{}] is not available." .format(train_function))
        exit(0)

    # Accuracy operation
    y = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, train_op, cross_entropy_loss, accuracy

# **********************************************************************************************************************
# FUNCTION: GET BATCHES FOR THE TRAIN OF THE NEURAL NETWORK
# **********************************************************************************************************************


def gen_batch_function(data_folder, image_shape):

    # Generate function to create batches of training data
    # :param data_folder: Path to folder that contains all the datasets
    # :param image_shape: Tuple - Shape of image
    # :return:

    def get_batches_fn(batch_size):

        # Create batches of training data
        # :param batch_size: Batch Size
        # :return: Batches of training data

        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # NO ENTIENDO ESTA PARTE DEL CODIGO
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

# **********************************************************************************************************************
# FUNCTION: TRAIN THE NEURAL NETWORK
# **********************************************************************************************************************


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
             keep_prob, learning_rate):
    # Train NN and print out the loss during training.
    # :param sess: TF Sesion
    # :param epochs: Number of epochs
    # :param batch_size: Batch size
    # :param get_batches_fn: Function to get batches of training data. Call using get_batches_fn(batch_size)
    # :param train_op: TF Operation to train the neural network
    # :param cross_entropy_loss: TF Tensor for the amount of loss
    # :param input_image: TF Placeholder for input images
    # :param correct_label: TF Placeholder for label images
    # :param keep_prob: TF Placeholder for dropout keep probability
    # :param learning_rate: TF Placeholder for learning rate

    sess.run(tf.global_variables_initializer())
    loss_per_epoch = []
    for epoch in range(epochs):
        losses, i = [], 0
        for images, labels in get_batches_fn(batch_size):
            i += 1
            feed_dict = {input_image: images,
                         correct_label: labels,
                         keep_prob: KEEP_PROB,
                         learning_rate: LEARNING_RATE}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            losses.append(loss)

        training_loss = sum(losses) / len(losses)
        loss_per_epoch.append(training_loss)
        if (epoch + 1) % 5 == 0:
            print(" [-] epoch: %d/%d, loss: %.5f" % (epoch + 1, epochs, training_loss))
    return loss_per_epoch

# **********************************************************************************************************************
# FUNCTION: TEST THE NEURAL NETWORK
# **********************************************************************************************************************


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):

    # Generate test output using the test images
    # :param sess: TF session
    # :param logits: TF Tensor for the logits
    # :param keep_prob: TF Placeholder for the dropout keep robability
    # :param image_pl: TF Placeholder for the image placeholder
    # :param data_folder: Path to the folder that contains the datasets
    # :param image_shape: Tuple - Shape of image
    # :return: Output for for each test image

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0,
                                                        image_pl: [image]})
        #NO ENTIENDO ESTO
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

        # RGBA mean: Red, Green, Blue , Alpha (transparency coefficient)
        # Generate a mask by multiplying by the green color
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        # Take the image (array) and convert it into a PIL image
        mask = scipy.misc.toimage(mask, mode="RGBA")

        # Take the image (array) and convert it into a PIL image
        street_im = scipy.misc.toimage(image)
        # Paste the mask in the image
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

# **********************************************************************************************************************
# FUNCTION: SAVES THE GENERATION OF SAMPLES
# **********************************************************************************************************************


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

# **********************************************************************************************************************
# CALL TO THE FUNCTIONS
# **********************************************************************************************************************

# Get layers of model VGG16
vgg_input, vgg_keep, vgg_layer3out, vgg_layer4out, vgg_layer7out = load_vgg(sess=, vgg_path=)

# Creation of the neural network layers from the modelVGG16
out = layers(vgg_layer3=vgg_layer3out, vgg_layer4=vgg_layer4out, vgg_layer7=vgg_layer7out, num_classes=NUM_CLASSES)

# Optimize the loss function to obtain the weights
logits, train_op, cross_entropy_loss, accuracy = optimize(nn_last_layer=out, correct_label=,
                                                          learning_rate=LEARNING_RATE, num_classes=NUM_CLASSES)

# Neural Network train
loss_epoch = train_nn(sess=, epochs=epochs, batch_size=BATCH_SIZE,
                      get_batches_fn=gen_batch_function(data_folder=, image_shape=IMAGE_SHAPE),
                      train_op=train_op, cross_entropy_loss=cross_entropy_loss, input_image=,
                      correct_label=, keep_prob=vgg_keep, learning_rate=LEARNING_RATE)

# Neural Network test
image_file, street_image = gen_test_output(sess=, logits=logits, keep_prob=vgg_keep,
                                           image_pl=, data_folder=, image_shape=IMAGE_SHAPE)

# Output of the test
save_inference_samples(runs_dir= RUNS_DIR, data_dir=DATA_DIR, sess=, image_shape=IMAGE_SHAPE,
                       logits=logits, keep_prob=vgg_keep, input_image=)
