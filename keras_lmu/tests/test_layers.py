# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf
from scipy.signal import cont2discrete

from keras_lmu import layers


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
def test_multivariate_lmu(rng, discretizer):
    memory_d = 4
    order = 16
    n_steps = 10 * order
    input_d = 32

    input_enc = rng.uniform(0, 1, size=(input_d, memory_d))

    # check that one multivariate LMU is the same as n one-dimensional LMUs (omitting
    # the hidden part)
    inp = tf.keras.Input(shape=(n_steps, input_d))
    multi_lmu = tf.keras.layers.RNN(
        layers.LMUCell(
            memory_d=memory_d,
            order=order,
            theta=n_steps,
            discretizer=discretizer,
            kernel_initializer=tf.initializers.constant(input_enc),
            hidden_cell=tf.keras.layers.SimpleRNNCell(
                units=memory_d * order,
                activation=None,
                kernel_initializer=tf.initializers.constant(np.eye(memory_d * order)),
                recurrent_initializer=tf.initializers.zeros(),
            ),
        ),
        return_sequences=True,
    )(inp)
    lmus = [
        tf.keras.layers.RNN(
            layers.LMUCell(
                memory_d=1,
                order=order,
                theta=n_steps,
                discretizer=discretizer,
                kernel_initializer=tf.initializers.constant(input_enc[:, [i]]),
                hidden_cell=tf.keras.layers.SimpleRNNCell(
                    units=order,
                    activation=None,
                    kernel_initializer=tf.initializers.constant(np.eye(order)),
                    recurrent_initializer=tf.initializers.zeros(),
                ),
            ),
            return_sequences=True,
        )(inp)
        for i in range(memory_d)
    ]

    model = tf.keras.Model(inp, [multi_lmu] + lmus)

    results = model.predict(rng.uniform(0, 1, size=(1, n_steps, input_d)))

    for i in range(memory_d):
        assert np.allclose(
            results[0][..., i * order : (i + 1) * order], results[i + 1], atol=2e-5
        ), np.max(abs(results[0][..., i * order : (i + 1) * order] - results[i + 1]))


@pytest.mark.parametrize("has_input_kernel", (True, False))
@pytest.mark.parametrize("fft", (True, False))
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
def test_layer_vs_cell(rng, has_input_kernel, fft, discretizer):
    n_steps = 10
    input_d = 32
    kwargs = dict(
        memory_d=4 if has_input_kernel else input_d,
        order=12,
        theta=n_steps * (8 if discretizer == "euler" else 1),
        discretizer=discretizer,
        kernel_initializer="glorot_uniform" if has_input_kernel else None,
        memory_to_memory=not fft,
    )
    hidden_cell = lambda: tf.keras.layers.SimpleRNNCell(units=64)

    inp = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    lmu_cell = tf.keras.layers.RNN(
        layers.LMUCell(hidden_cell=hidden_cell(), **kwargs),
        return_sequences=True,
    )
    cell_out = lmu_cell(inp)

    lmu_layer = layers.LMU(return_sequences=True, hidden_cell=hidden_cell(), **kwargs)
    lmu_layer.build(inp.shape)
    lmu_layer.layer.set_weights(lmu_cell.get_weights())
    layer_out = lmu_layer(inp)

    assert isinstance(lmu_layer.layer, layers.LMUFFT if fft else tf.keras.layers.RNN)

    for w0, w1 in zip(
        sorted(lmu_cell.weights, key=lambda w: w.shape.as_list()),
        sorted(lmu_layer.weights, key=lambda w: w.shape.as_list()),
    ):
        assert np.allclose(w0.numpy(), w1.numpy())

    assert np.allclose(cell_out, lmu_cell(inp))
    assert np.allclose(cell_out, layer_out, atol=3e-6 if fft else 1e-8), np.max(
        np.abs(cell_out - layer_out)
    )


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("trainable_theta", (True, False))
def test_save_load_weights(rng, tmp_path, discretizer, trainable_theta):
    memory_d = 4
    order = 12
    n_steps = 10
    input_d = 32

    x = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    inp = tf.keras.Input((None, input_d))
    lmu0 = layers.LMU(
        memory_d,
        order,
        n_steps,
        tf.keras.layers.SimpleRNNCell(units=64),
        discretizer=discretizer,
        trainable_theta=trainable_theta,
        return_sequences=True,
    )(inp)
    model0 = tf.keras.Model(inp, lmu0)
    out0 = model0(x)

    lmu1 = layers.LMU(
        memory_d,
        order,
        n_steps,
        tf.keras.layers.SimpleRNNCell(units=64),
        discretizer=discretizer,
        trainable_theta=trainable_theta,
        return_sequences=True,
    )(inp)
    model1 = tf.keras.Model(inp, lmu1)
    out1 = model1(x)

    assert not np.allclose(out0, out1)

    model0.save_weights(str(tmp_path))
    model1.load_weights(str(tmp_path))

    out2 = model1(x)
    assert np.allclose(out0, out2)


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("trainable_theta", (True, False))
@pytest.mark.parametrize("mode", ("cell", "lmu", "fft"))
def test_save_load_serialization(mode, tmp_path, trainable_theta, discretizer):
    if mode == "fft" and trainable_theta:
        pytest.skip("FFT does not support trainable theta")

    inp = tf.keras.Input((10 if mode == "fft" else None, 32))
    if mode == "cell":
        out = tf.keras.layers.RNN(
            layers.LMUCell(
                1,
                2,
                3,
                tf.keras.layers.SimpleRNNCell(4),
                trainable_theta=trainable_theta,
                discretizer=discretizer,
            ),
            return_sequences=True,
        )(inp)
    elif mode == "lmu":
        out = layers.LMU(
            1,
            2,
            3,
            tf.keras.layers.SimpleRNNCell(4),
            return_sequences=True,
            memory_to_memory=True,
            trainable_theta=trainable_theta,
            discretizer=discretizer,
        )(inp)
    elif mode == "fft":
        out = layers.LMUFFT(
            1,
            2,
            3,
            tf.keras.layers.SimpleRNNCell(4),
            discretizer=discretizer,
            return_sequences=True,
        )(inp)

    model = tf.keras.Model(inp, out)

    model.save(str(tmp_path))

    model_load = tf.keras.models.load_model(
        str(tmp_path),
        custom_objects={
            "LMUCell": layers.LMUCell,
            "LMU": layers.LMU,
            "LMUFFT": layers.LMUFFT,
        },
    )

    assert np.allclose(
        model.predict(np.ones((32, 10, 32))), model_load.predict(np.ones((32, 10, 32)))
    )


@pytest.mark.parametrize("return_sequences", (True, False))
@pytest.mark.parametrize(
    "hidden_cell",
    (
        lambda: None,
        lambda: tf.keras.layers.Dense(4),
        lambda: tf.keras.layers.SimpleRNNCell(4),
    ),
)
@pytest.mark.parametrize("memory_d", [1, 4])
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
def test_fft(return_sequences, hidden_cell, memory_d, discretizer, rng):
    kwargs = dict(
        memory_d=memory_d,
        order=2,
        theta=12,
        hidden_cell=hidden_cell(),
        discretizer=discretizer,
    )

    x = rng.uniform(-1, 1, size=(2, 10, 32))

    rnn_layer = tf.keras.layers.RNN(
        layers.LMUCell(**kwargs),
        return_sequences=return_sequences,
    )
    rnn_out = rnn_layer(x)

    fft_layer = layers.LMUFFT(return_sequences=return_sequences, **kwargs)
    fft_layer.build(x.shape)
    fft_layer.kernel.assign(rnn_layer.cell.kernel)
    fft_out = fft_layer(x, training=None)

    assert np.allclose(rnn_out, fft_out, atol=2e-6)


def test_validation_errors():
    fft_layer = layers.LMUFFT(1, 2, 3, None)
    with pytest.raises(ValueError, match="temporal axis be fully specified"):
        fft_layer(tf.keras.Input((None, 32)))

    with pytest.raises(ValueError, match="hidden_to_memory must be False"):
        layers.LMUCell(1, 2, 3, None, hidden_to_memory=True)

    with pytest.raises(ValueError, match="input_to_hidden must be False"):
        layers.LMUCell(1, 2, 3, None, input_to_hidden=True)

    with pytest.raises(ValueError, match="input_to_hidden must be False"):
        layers.LMUFFT(1, 2, 3, None, input_to_hidden=True)


@pytest.mark.parametrize(
    "should_use_fft, hidden_to_memory, memory_to_memory, steps, trainable_theta",
    [
        (True, False, False, 5, False),
        (False, True, False, 5, False),
        (False, False, True, 5, False),
        (False, False, False, None, False),
        (False, False, False, 5, True),
    ],
)
def test_fft_auto_swap(
    should_use_fft, hidden_to_memory, memory_to_memory, steps, trainable_theta
):
    lmu = layers.LMU(
        4,
        2,
        3,
        tf.keras.layers.Dense(5),
        hidden_to_memory=hidden_to_memory,
        memory_to_memory=memory_to_memory,
        trainable_theta=trainable_theta,
    )
    lmu.build((32, steps, 8))

    assert isinstance(lmu.layer, layers.LMUFFT) == should_use_fft


@pytest.mark.parametrize(
    "hidden_cell",
    (tf.keras.layers.SimpleRNNCell(units=10), tf.keras.layers.Dense(units=10), None),
)
@pytest.mark.parametrize("fft", (True, False))
def test_hidden_types(hidden_cell, fft, rng, seed):
    x = rng.uniform(-1, 1, size=(2, 5, 32))

    lmu_params = dict(
        memory_d=1,
        order=3,
        theta=4,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
    )

    base_lmu = tf.keras.layers.RNN(
        layers.LMUCell(hidden_cell=None, **lmu_params),
        return_sequences=True,
    )
    base_output = base_lmu(x)
    if isinstance(hidden_cell, tf.keras.layers.SimpleRNNCell):
        base_output = tf.keras.layers.RNN(hidden_cell, return_sequences=True)(
            base_output
        )
    elif isinstance(hidden_cell, tf.keras.layers.Dense):
        base_output = hidden_cell(base_output)

    lmu = (
        layers.LMUFFT(hidden_cell=hidden_cell, return_sequences=True, **lmu_params)
        if fft
        else tf.keras.layers.RNN(
            layers.LMUCell(hidden_cell=hidden_cell, **lmu_params),
            return_sequences=True,
        )
    )
    lmu_output = lmu(x)

    assert np.allclose(lmu_output, base_output, atol=2e-6 if fft else 1e-8)


@pytest.mark.parametrize("fft", (True, False))
@pytest.mark.parametrize("hidden_cell", (None, tf.keras.layers.Dense))
def test_connection_params(fft, hidden_cell):
    input_shape = (32, 7 if fft else None, 6)

    x = tf.keras.Input(batch_shape=input_shape)

    lmu_args = dict(
        memory_d=1,
        order=3,
        theta=4,
        hidden_cell=hidden_cell if hidden_cell is None else hidden_cell(units=5),
        input_to_hidden=False,
    )
    if not fft:
        lmu_args["hidden_to_memory"] = False
        lmu_args["memory_to_memory"] = False

    lmu = layers.LMUCell(**lmu_args) if not fft else layers.LMUFFT(**lmu_args)
    y = lmu(x) if fft else tf.keras.layers.RNN(lmu)(x)
    assert lmu.kernel.shape == (input_shape[-1], lmu.memory_d)
    if not fft:
        assert lmu.recurrent_kernel is None
    if hidden_cell is not None:
        assert lmu.hidden_cell.kernel.shape == (
            lmu.memory_d * lmu.order,
            lmu.hidden_cell.units,
        )
    assert y.shape == (
        input_shape[0],
        lmu.memory_d * lmu.order if hidden_cell is None else lmu.hidden_cell.units,
    )

    lmu_args["input_to_hidden"] = hidden_cell is not None
    if not fft:
        lmu_args["hidden_to_memory"] = hidden_cell is not None
        lmu_args["memory_to_memory"] = True

    lmu = layers.LMUCell(**lmu_args) if not fft else layers.LMUFFT(**lmu_args)
    if hidden_cell is not None:
        lmu.hidden_cell.built = False  # so that the kernel will be rebuilt
    y = lmu(x) if fft else tf.keras.layers.RNN(lmu)(x)
    assert lmu.kernel.shape == (
        input_shape[-1] + (0 if fft or hidden_cell is None else lmu.hidden_cell.units),
        lmu.memory_d,
    )
    if not fft:
        assert lmu.recurrent_kernel.shape == (
            lmu.order * lmu.memory_d,
            lmu.memory_d,
        )
    if hidden_cell is not None:
        assert lmu.hidden_cell.kernel.shape == (
            lmu.memory_d * lmu.order + input_shape[-1],
            lmu.hidden_cell.units,
        )
    assert y.shape == (
        input_shape[0],
        lmu.memory_d * lmu.order if hidden_cell is None else lmu.hidden_cell.units,
    )


@pytest.mark.parametrize(
    "dropout, recurrent_dropout, hidden_dropout, hidden_recurrent_dropout",
    [(0, 0, 0, 0), (0.5, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, 0.5, 0), (0, 0, 0, 0.5)],
)
@pytest.mark.parametrize("fft", (True, False))
def test_dropout(
    dropout, recurrent_dropout, hidden_dropout, hidden_recurrent_dropout, fft
):
    if fft:
        kwargs = {}
    else:
        kwargs = dict(memory_to_memory=True, recurrent_dropout=recurrent_dropout)
    lmu = layers.LMU(
        memory_d=1,
        order=3,
        theta=4,
        hidden_cell=tf.keras.layers.SimpleRNNCell(
            5, dropout=hidden_dropout, recurrent_dropout=hidden_recurrent_dropout
        ),
        dropout=dropout,
        **kwargs,
    )

    y0 = lmu(np.ones((32, 10, 64)), training=True).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=True).numpy()

    # if dropout is being applied then outputs should be stochastic, else deterministic
    assert np.allclose(y0, y1) != (
        dropout > 0
        or (recurrent_dropout > 0 and not fft)
        or hidden_dropout > 0
        or hidden_recurrent_dropout > 0
    )

    # dropout not applied when training=False
    y0 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    assert np.allclose(y0, y1)


@pytest.mark.parametrize("trainable_theta", (True, False))
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("fft", (True, False))
def test_fit(fft, discretizer, trainable_theta):
    if fft and trainable_theta:
        pytest.skip("FFT does not support trainable theta")

    lmu_layer = layers.LMU(
        memory_d=1,
        order=256,
        theta=784 if discretizer == "zoh" else 2000,
        trainable_theta=trainable_theta,
        hidden_cell=tf.keras.layers.SimpleRNNCell(units=30),
        hidden_to_memory=not fft,
        memory_to_memory=not fft,
        input_to_hidden=not fft,
        discretizer=discretizer,
        kernel_initializer="zeros",
    )

    inputs = tf.keras.layers.Input((5 if fft else None, 10))
    lmu = lmu_layer(inputs)
    outputs = tf.keras.layers.Dense(2)(lmu)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    x_train = tf.ones((5, 5, 10))
    x_test = tf.ones((5, 5, 10))
    y_train = tf.ones((5, 1))
    y_test = tf.ones((5, 1))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    _, acc = model.evaluate(x_test, y_test, verbose=0)

    assert isinstance(lmu_layer.layer, layers.LMUFFT if fft else tf.keras.layers.RNN)
    assert acc == 1.0


@pytest.mark.parametrize("fft", (True, False))
def test_no_input_kernel_dimension_mismatch(fft):
    lmu_layer = layers.LMU(
        memory_d=1,
        order=4,
        theta=4,
        hidden_cell=tf.keras.layers.SimpleRNNCell(units=10),
        hidden_to_memory=False,
        memory_to_memory=not fft,
        input_to_hidden=not fft,
        kernel_initializer=None,
    )

    with pytest.raises(ValueError, match="no input kernel"):
        lmu_layer(tf.ones((4, 10, 2)))


def test_discretizer_types():
    with pytest.raises(ValueError, match="discretizer must be 'zoh' or 'euler'"):
        layers.LMUCell(
            memory_d=1, order=256, theta=784, hidden_cell=None, discretizer="test"
        )


@pytest.mark.parametrize("trainable_theta", (True, False))
def test_discretizer_equivalence(trainable_theta, rng):
    # check that zoh and euler produce approximately the same output
    layer_zoh = layers.LMU(
        memory_d=2,
        order=8,
        theta=256,
        hidden_cell=None,
        discretizer="zoh",
        return_sequences=True,
        kernel_initializer=None,
        trainable_theta=trainable_theta,
    )
    layer_euler = layers.LMU(
        memory_d=2,
        order=8,
        theta=256,
        hidden_cell=None,
        discretizer="euler",
        return_sequences=True,
        kernel_initializer=None,
        trainable_theta=trainable_theta,
    )

    x = rng.uniform(-1, 1, size=(32, 10, 2))

    zoh = layer_zoh(x)
    euler = layer_euler(x)

    assert np.allclose(zoh, euler, atol=0.02), np.max(np.abs(zoh - euler))


def test_cont2discrete_zoh(rng):
    A = rng.randn(64, 64)
    B = rng.randn(64, 1)
    C = np.ones((1, 64))
    D = np.zeros((1,))

    scipy_A, scipy_B, *_ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    tf_A, tf_B = layers.LMUCell._cont2discrete_zoh(A.T, B.T)

    assert np.allclose(scipy_A, tf.transpose(tf_A))
    assert np.allclose(scipy_B, tf.transpose(tf_B))


@pytest.mark.parametrize("discretizer", ("euler", "zoh"))
@pytest.mark.parametrize("trainable_theta", (True, False))
def test_theta_update(discretizer, trainable_theta, tmp_path):
    # create model
    theta = 10
    lmu_cell = layers.LMUCell(
        memory_d=2,
        order=3,
        theta=theta,
        trainable_theta=trainable_theta,
        hidden_cell=tf.keras.layers.SimpleRNNCell(units=4),
        discretizer=discretizer,
    )

    inputs = tf.keras.layers.Input((None, 20))
    lmu = tf.keras.layers.RNN(lmu_cell)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=lmu)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam()
    )

    # make sure theta_inv is set correctly to initial value
    assert np.allclose(lmu_cell.theta_inv.numpy(), 1 / theta)

    # fit model on some data
    model.fit(tf.ones((64, 5, 20)), tf.ones((64, 4)), epochs=1)

    # make sure theta kernel got updated if trained
    assert np.allclose(lmu_cell.theta_inv.numpy(), 1 / theta) != trainable_theta

    # save model and make sure you get same outputs, that is, correct theta was stored
    model.save(str(tmp_path))

    model_load = tf.keras.models.load_model(
        str(tmp_path), custom_objects={"LMUCell": layers.LMUCell}
    )

    assert np.allclose(
        model.predict(np.ones((32, 10, 20))),
        model_load.predict(np.ones((32, 10, 20))),
    )


@pytest.mark.parametrize("mode", ("cell", "rnn", "fft"))
def test_theta_attribute(mode):
    theta = 3

    # check LMUCell theta attribute
    if mode == "cell":
        layer = layers.LMUCell(1, 2, theta, None, trainable_theta=True)
    elif mode == "rnn":
        layer = layers.LMU(1, 2, theta, None, trainable_theta=True)
    elif mode == "fft":
        layer = layers.LMU(1, 2, theta, None, trainable_theta=False)

    assert not layer.built
    assert layer.theta == theta

    layer.build((1, 1))
    assert layer.built
    assert np.allclose(layer.theta, theta)

    if mode == "fft":
        # fft doesn't support trainable theta
        assert isinstance(layer.layer, layers.LMUFFT)
    else:
        # check that updates to the internal variable are reflected in the theta
        # attribute
        cell = layer if mode == "cell" else layer.layer.cell
        cell.theta_inv.assign(10)
        assert np.allclose(layer.theta, 0.1)
