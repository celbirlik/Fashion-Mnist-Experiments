
import six
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, LeakyReLU, Layer, ReLU, Lambda, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import get_source_inputs
import numpy as np


def _bn_relu(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault(
        "kernel_regularizer", l2(1.e-4))

    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation
    function. This is the ResNet v2 scheme proposed in
    http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault(
        "kernel_regularizer", l2(1.e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(
        round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = BatchNormalization(axis=CHANNEL_AXIS,
                                      name=bn_name_base)(shortcut)

    return Add()([shortcut, residual])


def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None, transition_dilation_rates=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv):
    """Builds a residual block with repeating bottleneck blocks.
       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating
            layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each
            transition
    """
    if transition_dilation_rates is None:
        transition_dilation_rates = [(1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
        return x

    return f


def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by
    stage and block.
    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the
    paper and keras and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base


def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
                residual_unit=_bn_relu_conv):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)

    return f


def bottleneck(filters, stage, block, transition_strides=(1, 1),
               dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
               residual_unit=_bn_relu_conv):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input_feature):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(1, 1),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_feature)
        else:
            x = residual_unit(filters=filters, kernel_size=(1, 1),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_feature)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters * 4, kernel_size=(1, 1),
                          conv_name_base=conv_name_base + '2c',
                          bn_name_base=bn_name_base + '2c')(x)

        return _shortcut(input_feature, x)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _string_to_function(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def ResNet(input_shape=None, classes=10, block='bottleneck', residual_unit='v2',
           repetitions=None, initial_filters=64, activation='softmax', include_top=True,
           input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
           initial_strides=(1, 1), initial_kernel_size=(3, 3), initial_pooling=None,
           final_pooling=None, top='classification'):
    """Builds a custom ResNet like architecture. Defaults to ResNet50 v2.

    Returns:
        The keras `Model`.
    """
    if activation not in ['softmax', 'sigmoid', None]:
        raise ValueError(
            'activation must be one of "softmax", "sigmoid", or None')
    if activation == 'sigmoid' and classes != 1:
        raise ValueError(
            'sigmoid activation can only be used when classes = 1')
    if repetitions is None:
        repetitions = [3, 4, 6, 3]
    # Determine proper input shape

    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception(
            "Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    if block == 'basic':
        block_fn = basic_block
    elif block == 'bottleneck':
        block_fn = bottleneck
    elif isinstance(block, six.string_types):
        block_fn = _string_to_function(block)
    else:
        block_fn = block

    if residual_unit == 'v2':
        residual_unit = _bn_relu_conv
    elif residual_unit == 'v1':
        residual_unit = _conv_bn_relu
    elif isinstance(residual_unit, six.string_types):
        residual_unit = _string_to_function(residual_unit)
    else:
        residual_unit = residual_unit

    # Permute dimension order if necessary
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])
    # Determine proper input shape

    img_input = Input(shape=input_shape, tensor=input_tensor)
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=initial_strides)(img_input)
    if initial_pooling == 'max':
        x = MaxPooling2D(pool_size=(
            3, 3), strides=initial_strides, padding="same")(x)

    block = x
    filters = initial_filters
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)

    # Classifier block
    if include_top and top is 'classification':
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=classes, activation=activation,
                  kernel_initializer="he_normal")(x)
    elif include_top and top is 'segmentation':
        x = Conv2D(classes, (1, 1), activation='linear', padding='same')(x)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, classes))(x)
    elif final_pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif final_pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(inputs=img_input, outputs=x)
    return model


def ResNet18(input_shape=(28, 28, 1), classes=10, dropout=None):
    """ResNet with 18 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[2, 2, 2, 2], dropout=dropout)


def ResNet34(input_shape=(28, 28, 1), classes=10, dropout=None):
    """ResNet with 34 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[3, 4, 6, 3], dropout=dropout)


def ResNet50(input_shape=(28, 28, 1), classes=10):
    """ResNet with 50 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 4, 6, 3])


def ResNet101(input_shape, classes):
    """ResNet with 101 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 4, 23, 3])


def ResNet152(input_shape, classes):
    """ResNet with 152 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 8, 36, 3])


def initial_conv(input):
    x = Conv2D(16, 3, 3, border_mode='same')(input)

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 16 * k, else create Conv2D for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 16 * k:
            init = Conv2D(
                16 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 16 * k:
            init = Conv2D(
                16 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Conv2D(16 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(16 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m


def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create Conv2D for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 32 * k:
            init = Conv2D(
                32 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Conv2D(
                32 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Conv2D(32 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(32 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m


def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 64 * k, else create Conv2D for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 64 * k:
            init = Conv2D(
                64 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Conv2D(
                64 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Conv2D(64 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(64 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m


def create_WRN(input_dim=(28, 28, 1), nb_classes=10, N=2, k=4, dropout=0.0, verbose=0):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose:
        print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


class ShakeShake(Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = K.shape(x1)[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        beta = K.random_uniform((batch_size, 1, 1, 1))
        # shake-shake during training phase

        def x_shake():
            return beta * x1 + (1 - beta) * x2 + K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
        # even-even during testing phase

        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return K.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


def create_residual_branch(x, filters, stride):
    """ Regular Branch of a Residual network: ReLU -> Conv2D -> BN repeated twice """
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def create_residual_shortcut(x, filters, stride):
    """ Shortcut Branch used when downsampling from Shake-Shake regularization """
    x = ReLU()(x)
    x1 = Lambda(lambda y: y[:, 0:-1:stride, 0:-1:stride, :])(x)
    x1 = Conv2D(filters // 2, kernel_size=1, strides=1, padding='valid',
                kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                use_bias=False)(x1)
    x2 = Lambda(lambda y: y[:, 1::stride, 1::stride, :])(x)
    x2 = Conv2D(filters // 2, kernel_size=1, strides=1, padding='valid',
                kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                use_bias=False)(x2)
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    return x


def create_residual_block(x, filters, stride=1):
    """ Residual Block with Shake-Shake regularization and shortcut """
    x1 = create_residual_branch(x, filters, stride)
    x2 = create_residual_branch(x, filters, stride)
    if stride > 1:
        x = create_residual_shortcut(x, filters, stride)
    return Add()([x, ShakeShake()([x1, x2])])


def create_residual_layer(x, filters, blocks, stride):
    """ Layer repeating Residual Blocks """
    x = create_residual_block(x, filters, stride)
    for i in range(1, blocks):
        x = create_residual_block(x, filters, 1)
    return x


def create_shakeshake_res34(n_classes=10, n_blocks=[5, 5, 5], input_shape=(32, 32, 1), activation='softmax'):
    """ Residual Network with Shake-Shake regularization modeled after ResNet32 """
    # Input and first convolutional layer
    x_in = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
               use_bias=False)(x_in)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # Three stages of four residual blocks
    x = create_residual_layer(x, 64, n_blocks[0], 1)
    x = create_residual_layer(x, 128, n_blocks[1], 2)
    x = create_residual_layer(x, 256, n_blocks[2], 2)
    x = create_residual_layer(x, 512, n_blocks[3], 2)
    # Output pooling and dense layer
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x_out = Dense(n_classes, activation=activation,
                  kernel_initializer='he_normal')(x)
    return Model(x_in, x_out)


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip],
                   name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1,
               padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides,
                        padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1,
               padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(
        axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(
            axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                             name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(
            axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1,
                    padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(
            axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(
            axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle,
                 name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1+i))

    return x


def create_ShufflenetV2(include_top=True,
                        input_tensor=None,
                        scale_factor=1.0,
                        pooling='avg',
                        input_shape=(28, 28, 1),
                        load_model=None,
                        num_shuffle_units=[3, 7, 3],
                        bottleneck_ratio=1,
                        classes=10):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(
        scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units),
                              dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    # calculate output channels for each stage
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                  padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1,
               name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model


def create_miniVGG(classes=10, input_shape=(28, 28, 1)):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding="same", activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(LeakyReLU())
    # model2.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model


def create_basic_cnn(classes=10, input_shape=(28, 28, 1)):

    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=1, padding='same')(input)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.35)(x)
    x = Flatten()(x)
    x = Dense(200)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(classes)(x)
    x = Activation('softmax')(x)
    return Model(inputs=input, outputs=x)
