from keras.layers.convolutional import Conv2D
from keras.utils import conv_utils
from keras import backend as K


# For now, we introduce support for the improved implementation only for the `tensorflow` backend
if K.backend() == "tensorflow":
    def get_image_patches(images, psize):
        return K.tf.extract_image_patches(images, (1,) + psize + (1,), (1, 1, 1, 1), (1, 1, 1, 1), 'VALID')

    K.get_image_patches = get_image_patches


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

    def call(self, inputs):
        patches = K.get_image_patches(inputs, self.effective_kernel_size)
        input_shape = K.get_variable_shape(inputs)
        if self.data_format == 'channels_first':
            n_ch = input_shape[0]
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            n_ch = input_shape[3]

        # After the reshape, patches will be with shape:
        # `(batch_size * num_patches, effective_kernel_rows, effective_kernel_cols, n_in_channels)`
        patches = K.reshape(
            patches,
            (-1, self.effective_kernel_size[0], self.effective_kernel_size[1], n_ch)
        )
        # After permute_dimensions, patches will be with shape:
        # `(effective_kernel_rows, effective_kernel_cols, n_in_channels, batch_size * num_patches)`
        patches = K.permute_dimensions(patches, (1, 2, 3, 0))

        meta_kernel_h, meta_kernel_w, input_depth, n_filters = K.get_variable_shape(self.kernel)
        meta_kernel = K.permute_dimensions(self.kernel, (3, 0, 1, 2))  # make n_filters first dimension

        # Now meta_kernel acts as image and the patches as filters.
        # result will be of shape `(n_filters, 1, offset_h, offset_w, batch_size)`
        offset_h = self.kernel_size[0] - self.effective_kernel_size[0] + 1
        offset_w = self.kernel_size[1] - self.effective_kernel_size[1] + 1
        outputs = K.conv2d(meta_kernel, patches, data_format='channels_last')
        outputs = K.permute_dimensions(outputs, (3, 1, 2, 0))
        #  Outputs shape is now `(batch_size * patches, n_filters, offset_h, offset_w)`

        outputs = K.pool2d(outputs, self.pool_size, data_format='channels_last')
        num_patches_h = rows - self.effective_kernel_size[0] + 1
        num_patches_w = cols - self.effective_kernel_size[1] + 1
        outputs = K.reshape(outputs, (-1, num_patches_h, num_patches_w,
                                      n_filters * (offset_h // self.pool_size[0]) * (offset_w // self.pool_size[1])
                                      )
                            )

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

