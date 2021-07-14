# pylint: disable=missing-docstring

import timeit

import numpy as np
import pytest
import tensorflow as tf

from keras_lmu import layers


class SteptimeLogger(tf.keras.callbacks.Callback):
    """Callback that records step times."""

    def __init__(self, count_mode="samples", stateful_metrics=None):
        super().__init__()

        self.train_start = None
        self.epoch_start = None
        self.batch_start = None

        self.train_time = None
        self.epoch_times = []
        self.batch_times = []

    def on_train_begin(self, logs=None):
        self.train_start = timeit.default_timer()

    def on_train_end(self, logs=None):
        self.train_time = timeit.default_timer() - self.train_start

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = timeit.default_timer()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(timeit.default_timer() - self.epoch_start)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = timeit.default_timer()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_times.append(timeit.default_timer() - self.batch_start)


@pytest.mark.parametrize("mode", ["rnn", "fft", "raw"])
def test_performance(mode, capsys):
    dims = 32
    seq_len = 1024
    batch_size = 4
    odims = 2

    kwargs = dict(memory_d=dims, order=256, theta=784, hidden_cell=None)
    if mode == "rnn":
        lmu_layer = tf.keras.layers.RNN(
            layers.LMUCell(**kwargs),
            return_sequences=True,
        )
    elif mode in ("fft", "raw", "raw_nchw"):
        lmu_layer = layers.LMUFFT(
            return_sequences=True, conv_mode=mode, truncate_ir=None, **kwargs
        )

    inputs = tf.keras.layers.Input((seq_len, dims), batch_size=batch_size)
    lmu = lmu_layer(inputs)
    outputs = tf.keras.layers.Dense(odims)(lmu)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    n_train = 5 * batch_size
    x_train = tf.random.uniform((n_train, seq_len, dims), minval=-1, maxval=1, seed=0)
    y_train = tf.random.uniform((n_train, seq_len, odims), minval=-1, maxval=1, seed=1)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.RMSprop(),
    )

    steptimes = SteptimeLogger()
    model.fit(x_train, y_train, epochs=2, batch_size=batch_size, callbacks=[steptimes])

    step_time = np.mean(steptimes.batch_times[1:])

    with capsys.disabled():
        print(f"step time: {step_time:0.3f}")
