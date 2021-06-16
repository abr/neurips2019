.. _basic-usage:

***********
Basic usage
***********

The standard Legendre Memory Unit (LMU) layer
implementation in KerasLMU is defined in the
``keras_lmu.LMU`` class. The following code creates
a new LMU layer:

.. testcode::

   import keras_lmu

   lmu_layer = keras_lmu.LMU(
       memory_d=1,
       order=256,
       theta=784,
       hidden_cell=tf.keras.layers.SimpleRNNCell(units=10),
   )

Note that the values used above for ``memory_d``, ``order``,
``theta``, and ``units`` are arbitrary example values; actual parameter settings will
depend on your specific application.
``memory_d`` represents the dimensionality of the signal represented in the LMU memory,
``order`` represents the dimensionality of the LMU basis,
``theta`` represents the dimensionality of the sliding window,
and ``units`` represents the dimensionality of the hidden component.
To learn more about these parameters, check out
the :ref:`LMU class API reference <api-reference-lc>`.

Creating KerasLMU layers
------------------------

The ``LMU`` class functions as a standard
Keras layer and is meant to be used within a Keras model.
The code below illustrates how to do this using a Keras model with
a 10-dimensional input and a 20-dimensional output.

.. testcode::

   from tensorflow.keras import Input, Model
   from tensorflow.keras.layers import Dense

   inputs = Input((None, 10))
   lmus = lmu_layer(inputs)
   outputs = Dense(20)(lmus)

   model = Model(inputs=inputs, outputs=outputs)


Other parameters
----------------

The ``LMU`` class has several other configuration options; see
:doc:`the API reference <api-reference>` for all the details.
