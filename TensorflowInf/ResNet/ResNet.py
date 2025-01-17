'''

Authors: Nishant Kumar.

Copyright:
Copyright (c) 2018 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Parts of the code in this file was taken from the original model from 
https://github.com/tensorflow/models/tree/master/official/r1/resnet.

'''

import os, sys
import time
import numpy as np
import argparse
import tensorflow as tf
import ResNet_Model

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "TFCompiler"))
import DumpTFMtData
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ImgOps"))
from ImgOps import getImg, squareCrop, imgResize, reverseColor, meanOffset

NUM_CLASSES = 1001

##############################################
# Model related functions

def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
        resnet_size: The number of convolutional layers needed in the model.

    Returns:
        A list of block sizes to use in building the model.

    Raises:
        KeyError: if invalid resnet_size is received.
    """
    choices = {
        1: [0, 0, 0, 0],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
                'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
        raise ValueError(err)

class ImagenetModel(ResNet_Model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self,
        resnet_size,
        data_format = None,
        num_classes = NUM_CLASSES,
        resnet_version = ResNet_Model.DEFAULT_VERSION,
        dtype=ResNet_Model.DEFAULT_DTYPE):

        """These are the parameters that work for Imagenet data.
        Args:
            resnet_size: The number of convolutional layers needed in the model.
                data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
                num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
                resnet_version: Integer representing which version of the ResNet network
            to use. See README for details. Valid values: [1, 2]
                dtype: The TensorFlow dtype to use for calculations.
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(ImagenetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )

##############################################

def infer(imgfile, scalingFac):
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
    # y = tf.placeholder(tf.int64, shape=(None), name='input_y')

    imgnet_model = ImagenetModel(50, 'channels_last')
    pred = imgnet_model(x, False)
    pred = tf.argmax(pred, 1)
    # correct_pred = tf.equal(numericPred, y)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = getImg(imgfile)
        img = squareCrop(img)
        img = imgResize(img, 224, 224)
        img = reverseColor(img)
        img = meanOffset(img)
        images = [img]

        numImages = len(images)
        print("lenimages = ", numImages)
        feed_dict = {x: images}

        output_tensor = None
        gg = tf.get_default_graph()
        for node in gg.as_graph_def().node:
            if node.name == 'ArgMax':
                output_tensor = gg.get_operation_by_name(node.name).outputs[0]

        optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)
        modelPath = './PreTrainedModel/resnet_v2_fp32_savedmodel_NHWC/1538687283/variables/variables'
        saver = tf.train.Saver()
        saver.restore(sess, modelPath)
        DumpTFMtData.updateWeightsForBN(optimized_graph_def, sess, feed_dict)

        trainVarsName = []
        for node in optimized_graph_def.node:
            if node.op=="VariableV2":
                trainVarsName.append(node.name)
        trainVars = list(map(lambda x : tf.get_default_graph().get_operation_by_name(x).outputs[0] , trainVarsName))
        DumpTFMtData.dumpImgAndWeightsData(sess, images[0], trainVars, 'ResNet_img_input.inp', scalingFac)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="SampleImages/n02109961_36.JPEG", help="input file")
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    infer(args.img, 12)

if __name__=='__main__':
    main()
