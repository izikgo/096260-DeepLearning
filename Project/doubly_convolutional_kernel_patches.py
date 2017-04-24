from keras.layers.convolutional import Conv2D
from keras.utils import conv_utils
from keras import backend as K


class DoubleConv2D(Conv2D):
    """Double 2D convolution.

    # Arguments
        filters: Integer, the number of meta-filters to create
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution meta-window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        effective_kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions. The size must be contained in `kernel_size`.
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale the output of the convolution with the
            meta-filters. The result is flattened and used as the different channels
            values. If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, width, height, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, width, height)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 effective_kernel_size,
                 pool_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DoubleConv2D, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.effective_kernel_size = effective_kernel_size
        self.pool_size = pool_size

    def get_effective_kernel(self):
        meta_kernel_h, meta_kernel_w, input_depth, n_filters = K.get_variable_shape(self.kernel)
        effective_kernel_h, effective_kernel_w = self.effective_kernel_size
        meta_kernel = K.permute_dimensions(self.kernel, (3, 0, 1, 2))  # make n_filters first dimension
        I = K.eye(input_depth * effective_kernel_h * effective_kernel_w)
        I = K.reshape(I, (effective_kernel_h, effective_kernel_w, input_depth,
                      input_depth * effective_kernel_h * effective_kernel_w))

        effective_kernel = K.conv2d(meta_kernel, I, data_format='channels_last')
        offset_h = meta_kernel_h - effective_kernel_h + 1
        offset_w = meta_kernel_w - effective_kernel_w + 1
        # shape is now (n_filters, offset_h, offset_w, input_depth * effective_kernel_h * effective_kernel_w)

        effective_kernel = K.reshape(effective_kernel, ((n_filters * offset_h * offset_w,
                                                         input_depth, effective_kernel_h, effective_kernel_w)))
        effective_kernel = K.permute_dimensions(effective_kernel, (2, 3, 1, 0))

        return effective_kernel

    def call(self, inputs):
        effective_kernel = self.get_effective_kernel()

        if self.data_format == 'channels_first':
            batch_size, n_filters, rows, cols = self.compute_output_shape(K.get_variable_shape(inputs))
        elif self.data_format == 'channels_last':
            batch_size, rows, cols, n_filters = self.compute_output_shape(K.get_variable_shape(inputs))
            inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))  # change to channels first

        outputs = K.conv2d(
            inputs,
            effective_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format='channels_first',
            dilation_rate=self.dilation_rate)

        offset_h = self.kernel_size[0] - self.effective_kernel_size[0] + 1
        offset_w = self.kernel_size[1] - self.effective_kernel_size[1] + 1

        outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))

        outputs = K.reshape(outputs, (-1, rows * cols * self.filters, offset_h, offset_w))

        outputs = K.pool2d(outputs, self.pool_size, data_format='channels_first')

        outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))
        outputs = K.reshape(outputs, (-1, n_filters, rows, cols))

        if self.data_format == 'channels_last':
            outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))  # return to channels last

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.effective_kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.effective_kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        offset_h = self.kernel_size[0] - self.effective_kernel_size[0] + 1
        offset_w = self.kernel_size[1] - self.effective_kernel_size[1] + 1
        n_filters = self.filters * (offset_h // self.pool_size[0]) * (offset_w // self.pool_size[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], n_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, n_filters)

    def get_config(self):
        config = super(DoubleConv2D, self).get_config()
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        return config

