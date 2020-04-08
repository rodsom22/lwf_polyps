"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras
from keras.utils import get_file

from . import retinanet
from . import Backbone
from retinanet.utils.image import preprocess_image


class InceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return inc_retinanet(*args, backbone=self.backbone, **kwargs)
    
    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inc']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def inc_retinanet(num_classes, backbone='inc', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a inception resnet v2 backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (always 'inc').
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a Inception Resnet v2
         backbone.
    """
    # choose default input

    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the inc backbone
    inc = keras.applications.inception_resnet_v2.InceptionResNetV2(input_tensor=inputs, include_top=False)

    if modifier:
        inc = modifier(inc)

    # create the full model
    layer_names = ["mixed_5b", "block17_20_conv", "conv_7b"]
    layer_outputs = [inc.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
