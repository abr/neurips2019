"""
Core classes for the KerasLMU package.
"""

import numpy as np
import tensorflow as tf
from scipy.signal import cont2discrete
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin


class LMUCell(DropoutRNNCellMixin, tf.keras.layers.Layer):
    """
    Implementation of LMU cell (to be used within Keras RNN wrapper).

    In general, the LMU cell consists of two parts: a memory component (decomposing
    the input signal using Legendre polynomials as a basis), and a hidden component
    (learning nonlinear mappings from the memory component). [1]_ [2]_

    This class processes one step within the whole time sequence input. Use the ``LMU``
    class to create a recurrent Keras layer to process the whole sequence. Calling
    ``LMU()`` is equivalent to doing ``RNN(LMUCell())``.

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : int
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of
        prediction, however the entire sequence will still be processed in order for
        information to be projected to and from the hidden layer.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer/RNNCell implementing the hidden component.
    hidden_to_memory : bool
        If True, connect the output of the hidden component back to the memory
        component (default False).
    memory_to_memory : bool
        If True, add a learnable recurrent connection (in addition to the static
        Legendre system) to the memory component (default False).
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    recurrent_initializer : ``tf.initializers.Initializer``
        Initializer for ``memory_to_memory`` weights (if that connection is enabled).
    dropout : float
        Dropout rate on input connections.
    recurrent_dropout : float
        Dropout rate on ``memory_to_memory`` connection.

    References
    ----------
    .. [1] Voelker and Eliasmith (2018). Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells.
       Neural Computation, 30(3): 569-609.
    .. [2] Voelker and Eliasmith. "Methods and systems for implementing
       dynamic neural networks." U.S. Patent Application No. 15/243,223.
       Filing date: 2016-08-22.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=False,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        dropout=0,
        recurrent_dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.memory_d = memory_d
        self.order = order
        self.theta = theta
        self.hidden_cell = hidden_cell
        self.hidden_to_memory = hidden_to_memory
        self.memory_to_memory = memory_to_memory
        self.input_to_hidden = input_to_hidden
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.kernel = None
        self.recurrent_kernel = None
        self.A = None
        self.B = None

        if self.hidden_cell is None:
            for conn in ("hidden_to_memory", "input_to_hidden"):
                if getattr(self, conn):
                    raise ValueError(f"{conn} must be False if hidden_cell is None")

            self.hidden_output_size = self.memory_d * self.order
            self.hidden_state_size = []
        elif hasattr(self.hidden_cell, "state_size"):
            self.hidden_output_size = self.hidden_cell.output_size
            self.hidden_state_size = self.hidden_cell.state_size
        else:
            # TODO: support layers that don't have the `units` attribute
            self.hidden_output_size = self.hidden_cell.units
            self.hidden_state_size = [self.hidden_cell.units]

        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))
        self._A, self._B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        self.state_size = tf.nest.flatten(self.hidden_state_size) + [
            self.memory_d * self.order
        ]
        self.output_size = self.hidden_output_size

    def build(self, input_shape):
        """
        Builds the cell.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shape)

        enc_d = input_shape[-1]
        if self.hidden_to_memory:
            enc_d += self.hidden_output_size

        if self.kernel_initializer is not None:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(enc_d, self.memory_d),
                initializer=self.kernel_initializer,
            )
        else:
            self.kernel = None
            if enc_d != self.memory_d:
                raise ValueError(
                    f"For LMUCells with no input kernel, the input dimension ({enc_d})"
                    f" must equal `memory_d` ({self.memory_d})."
                )

        if self.memory_to_memory:
            self.recurrent_kernel = self.add_weight(
                name="recurrent_kernel",
                shape=(self.memory_d * self.order, self.memory_d),
                initializer=self.recurrent_initializer,
            )
        else:
            self.recurrent_kernel = None

        if self.hidden_cell is not None and not self.hidden_cell.built:
            hidden_input_d = self.memory_d * self.order
            if self.input_to_hidden:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.hidden_cell.name):
                self.hidden_cell.build((input_shape[0], hidden_input_d))

        self.A = self.add_weight(
            name="A",
            shape=(self.order, self.order),
            initializer=tf.initializers.constant(self._A.T),  # note: transposed
            trainable=False,
        )

        self.B = self.add_weight(
            name="B",
            shape=(1, self.order),  # system is SISO
            initializer=tf.initializers.constant(self._B.T),  # note: transposed
            trainable=False,
        )

    def call(self, inputs, states, training=None):
        """
        Apply this cell to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        states = tf.nest.flatten(states)

        # state for the hidden cell
        h = states[:-1]
        # state for the LMU memory
        m = states[-1]

        # compute memory input
        u_in = tf.concat((inputs, h[0]), axis=1) if self.hidden_to_memory else inputs
        if self.dropout > 0:
            u_in *= self.get_dropout_mask_for_cell(u_in, training)
        u = u_in if self.kernel is None else tf.matmul(u_in, self.kernel)

        if self.memory_to_memory:
            if self.recurrent_dropout > 0:
                # note: we don't apply dropout to the memory input, only
                # the recurrent kernel
                rec_m = m * self.get_recurrent_dropout_mask_for_cell(m, training)
            else:
                rec_m = m

            u += tf.matmul(rec_m, self.recurrent_kernel)

        # separate memory/order dimensions
        m = tf.reshape(m, (-1, self.memory_d, self.order))
        u = tf.expand_dims(u, -1)

        # update memory
        m = tf.matmul(m, self.A) + tf.matmul(u, self.B)

        # re-combine memory/order dimensions
        m = tf.reshape(m, (-1, self.memory_d * self.order))

        # apply hidden cell
        h_in = tf.concat((m, inputs), axis=1) if self.input_to_hidden else m

        if self.hidden_cell is None:
            o = h_in
            h = []
        elif hasattr(self.hidden_cell, "state_size"):
            o, h = self.hidden_cell(h_in, h, training=training)
        else:
            o = self.hidden_cell(h_in, training=training)
            h = [o]

        return o, h + [m]

    def reset_dropout_mask(self):
        """Reset dropout mask for memory and hidden components."""
        super().reset_dropout_mask()
        if isinstance(self.hidden_cell, DropoutRNNCellMixin):
            self.hidden_cell.reset_dropout_mask()

    def reset_recurrent_dropout_mask(self):
        """Reset recurrent dropout mask for memory and hidden components."""
        super().reset_recurrent_dropout_mask()
        if isinstance(self.hidden_cell, DropoutRNNCellMixin):
            self.hidden_cell.reset_recurrent_dropout_mask()

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            dict(
                memory_d=self.memory_d,
                order=self.order,
                theta=self.theta,
                hidden_cell=tf.keras.layers.serialize(self.hidden_cell),
                hidden_to_memory=self.hidden_to_memory,
                memory_to_memory=self.memory_to_memory,
                input_to_hidden=self.input_to_hidden,
                kernel_initializer=self.kernel_initializer,
                recurrent_initializer=self.recurrent_initializer,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = tf.keras.layers.deserialize(config["hidden_cell"])
        return super().from_config(config)


class LMU(tf.keras.layers.Layer):
    """
    A layer of trainable low-dimensional delay systems.

    Each unit buffers its encoded input
    by internally representing a low-dimensional
    (i.e., compressed) version of the sliding window.

    Nonlinear decodings of this representation,
    expressed by the A and B matrices, provide
    computations across the window, such as its
    derivative, energy, median value, etc ([1]_, [2]_).
    Note that these decoder matrices can span across
    all of the units of an input sequence.

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : int
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of
        prediction, however the entire sequence will still be processed in order for
        information to be projected to and from the hidden layer.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer/RNNCell implementing the hidden component.
    hidden_to_memory : bool
        If True, connect the output of the hidden component back to the memory
        component (default False).
    memory_to_memory : bool
        If True, add a learnable recurrent connection (in addition to the static
        Legendre system) to the memory component (default False).
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    recurrent_initializer : ``tf.initializers.Initializer``
        Initializer for ``memory_to_memory`` weights (if that connection is enabled).
    dropout : float
        Dropout rate on input connections.
    recurrent_dropout : float
        Dropout rate on ``memory_to_memory`` connection.
    return_sequences : bool, optional
        If True, return the full output sequence. Otherwise, return just the last
        output in the output sequence.

    References
    ----------
    .. [1] Voelker and Eliasmith (2018). Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells.
       Neural Computation, 30(3): 569-609.
    .. [2] Voelker and Eliasmith. "Methods and systems for implementing
       dynamic neural networks." U.S. Patent Application No. 15/243,223.
       Filing date: 2016-08-22.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=False,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        dropout=0,
        recurrent_dropout=0,
        return_sequences=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.memory_d = memory_d
        self.order = order
        self.theta = theta
        self.hidden_cell = hidden_cell
        self.hidden_to_memory = hidden_to_memory
        self.memory_to_memory = memory_to_memory
        self.input_to_hidden = input_to_hidden
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.layer = None

    def build(self, input_shapes):
        """
        Builds the layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shapes)

        if (
            not self.hidden_to_memory
            and not self.memory_to_memory
            and input_shapes[1] is not None
        ):
            self.layer = LMUFFT(
                memory_d=self.memory_d,
                order=self.order,
                theta=self.theta,
                hidden_cell=self.hidden_cell,
                input_to_hidden=self.input_to_hidden,
                kernel_initializer=self.kernel_initializer,
                dropout=self.dropout,
                return_sequences=self.return_sequences,
            )
        else:
            self.layer = tf.keras.layers.RNN(
                LMUCell(
                    memory_d=self.memory_d,
                    order=self.order,
                    theta=self.theta,
                    hidden_cell=self.hidden_cell,
                    hidden_to_memory=self.hidden_to_memory,
                    memory_to_memory=self.memory_to_memory,
                    input_to_hidden=self.input_to_hidden,
                    kernel_initializer=self.kernel_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                ),
                return_sequences=self.return_sequences,
            )

        self.layer.build(input_shapes)

    def call(self, inputs, training=None):
        """
        Apply this layer to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        return self.layer.call(inputs, training=training)

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            dict(
                memory_d=self.memory_d,
                order=self.order,
                theta=self.theta,
                hidden_cell=tf.keras.layers.serialize(self.hidden_cell),
                hidden_to_memory=self.hidden_to_memory,
                memory_to_memory=self.memory_to_memory,
                input_to_hidden=self.input_to_hidden,
                kernel_initializer=self.kernel_initializer,
                recurrent_initializer=self.recurrent_initializer,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=self.return_sequences,
            )
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = tf.keras.layers.deserialize(config["hidden_cell"])
        return super().from_config(config)


class LMUFFT(tf.keras.layers.Layer):
    """
    Layer class for the FFT variant of the LMU.

    This class assumes no recurrent connections are desired in the memory component.

    Produces the output of the delay system by evaluating the convolution of the input
    sequence with the impulse response from the LMU cell. The convolution operation is
    calculated using the fast Fourier transform (FFT).

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : int
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of
        prediction, however the entire sequence will still be processed in order for
        information to be projected to and from the hidden layer.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer implementing the hidden component.
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    dropout : float
        Dropout rate on input connections.
    return_sequences : bool, optional
        If True, return the full output sequence. Otherwise, return just the last
        output in the output sequence.
    conv_mode : "fft" or "raw"
        The method for performing the inpulse response convolution. "fft" uses FFT
        convolution (default). "raw" uses explicit convolution, which may be faster
        for particular models on particular hardware.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        input_to_hidden=False,
        kernel_initializer="glorot_uniform",
        dropout=0,
        return_sequences=False,
        conv_mode="fft",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if input_to_hidden and hidden_cell is None:
            raise ValueError("input_to_hidden must be False if hidden_cell is None")

        self.memory_d = memory_d
        self.order = order
        self.theta = theta
        self.hidden_cell = hidden_cell
        self.input_to_hidden = input_to_hidden
        self.kernel_initializer = kernel_initializer
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.conv_mode = conv_mode.lower()

        # create a standard LMUCell to generate the impulse response during `build`
        self.delay_layer = tf.keras.layers.RNN(
            LMUCell(
                memory_d=1,
                order=order,
                theta=theta,
                hidden_cell=None,
                input_to_hidden=False,
                hidden_to_memory=False,
                memory_to_memory=False,
                kernel_initializer=None,
                trainable=False,
            ),
            return_sequences=True,
        )

    def build(self, input_shape):
        """
        Builds the layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shape)

        seq_len = input_shape[1]
        enc_d = input_shape[-1]

        if seq_len is None:
            # TODO: we could dynamically run the impulse response for longer if
            #  needed using stateful=True
            raise ValueError(
                f"LMUFFT requires that the input shape's temporal axis be fully "
                f"specified (got {seq_len})"
            )

        impulse = tf.reshape(tf.eye(seq_len, 1), (1, -1, 1))

        self.impulse_response = tf.squeeze(self.delay_layer(impulse), axis=0)
        self.impulse_response_fft = tf.signal.rfft(
            tf.transpose(self.impulse_response),
            fft_length=[2 * seq_len],
        )

        if self.kernel_initializer is not None:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_shape[-1], self.memory_d),
                initializer=self.kernel_initializer,
            )
        else:
            self.kernel = None
            if enc_d != self.memory_d:
                raise ValueError(
                    f"For LMUCells with no input kernel, the input dimension ({enc_d})"
                    f" must equal `memory_d` ({self.memory_d})."
                )

        if self.hidden_cell is not None and not self.hidden_cell.built:
            hidden_input_d = self.memory_d * self.order
            if self.input_to_hidden:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.hidden_cell.name):
                self.hidden_cell.build((input_shape[0], hidden_input_d))

    def call(self, inputs, training=None):
        """
        Apply this layer to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self.dropout:
            inputs = tf.keras.layers.Dropout(
                self.dropout, noise_shape=(inputs.shape[0], 1) + inputs.shape[2:]
            )(inputs)

        # Apply input encoders
        u = (
            inputs
            if self.kernel is None
            else tf.matmul(inputs, self.kernel, name="input_encoder_mult")
        )

        if self.conv_mode == "fft":
            m = self._fft_convolution(u)
        elif self.conv_mode == "raw":
            m = self._raw_convolution(u)
        else:
            raise ValueError(f"Unrecognized conv mode '{self.conv_mode}'")

        # apply hidden cell
        h_in = tf.concat((m, inputs), axis=-1) if self.input_to_hidden else m

        if self.hidden_cell is None:
            h = h_in if self.return_sequences else h_in[:, -1]
        elif hasattr(self.hidden_cell, "state_size"):
            h = tf.keras.layers.RNN(
                self.hidden_cell, return_sequences=self.return_sequences
            )(h_in, training=training)
        else:
            if not self.return_sequences:
                # no point applying the hidden cell to the whole sequence
                h = self.hidden_cell(h_in[:, -1], training=training)
            else:
                h = tf.keras.layers.TimeDistributed(self.hidden_cell)(
                    h_in, training=training
                )

        return h

    def _fft_convolution(self, u):
        seq_len = tf.shape(u)[1]

        # FFT requires shape (batch, memory_d, timesteps)
        u = tf.transpose(u, perm=[0, 2, 1])

        # Pad sequences to avoid circular convolution
        # Perform the FFT
        fft_input = tf.signal.rfft(u, fft_length=[2 * seq_len], name="input_pad")

        # Elementwise product of FFT (with broadcasting)
        result = tf.expand_dims(fft_input, axis=-2) * self.impulse_response_fft

        # Inverse FFT
        m = tf.signal.irfft(result, fft_length=[2 * seq_len])[..., :seq_len]

        m = tf.reshape(m, (-1, self.order * self.memory_d, seq_len))

        return tf.transpose(m, perm=[0, 2, 1])

    def _raw_convolution(self, u):
        seq_len = tf.shape(u)[1]
        assert self.impulse_response.shape[1:] == (self.order,)

        u = tf.transpose(u, perm=[0, 2, 1])
        u = tf.reshape(u, (-1, 1, 1, seq_len))  # combine batch and memory_d dimensions
        filters = tf.reshape(self.impulse_response, (1, seq_len, 1, self.order))
        filters = filters[:, ::-1, :, :]
        padding = [[0, 0], [0, 0], [0, 0], [seq_len - 1, 0]]
        m = tf.nn.conv2d(u, filters, strides=1, data_format="NCHW", padding=padding)
        m = tf.reshape(m, (-1, self.memory_d * self.order, seq_len))

        return tf.transpose(m, perm=[0, 2, 1])

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            dict(
                memory_d=self.memory_d,
                order=self.order,
                theta=self.theta,
                hidden_cell=tf.keras.layers.serialize(self.hidden_cell),
                input_to_hidden=self.input_to_hidden,
                kernel_initializer=self.kernel_initializer,
                dropout=self.dropout,
                return_sequences=self.return_sequences,
                conv_mode=self.conv_mode,
            )
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = tf.keras.layers.deserialize(config["hidden_cell"])
        return super().from_config(config)
