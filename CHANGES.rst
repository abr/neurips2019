***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

0.4.0 (unreleased)
==================

**Added**

- Setting ``kernel_initializer=None`` now removes the dense input kernel. (`#40`_)
- The ``keras_lmu.LMUFFT`` layer now supports ``memory_d > 1``. ``keras_lmu.LMU`` now
  uses this implementation for all values of ``memory_d`` when feedforward conditions
  are satisfied (no hidden-to-memory or memory-to-memory connections,
  and the sequence length is not ``None``). (`#40`_)
- Added ``trainable_theta`` option, which will allow the ``theta`` parameter to be
  learned during training. (`#41`_)
- Added ``discretizer`` option, which controls the method used to solve for the ``A``
  and ``B`` LMU matrices. This is mainly useful in combination with
  ``trainable_theta=True``, where setting ``discretizer="euler"`` may improve the
  training speed (possibly at the cost of some accuracy). (`#41`_)

**Changed**

- The ``A`` and ``B`` matrices are now stored as constants instead of non-trainable
  variables. This can improve the training/inference speed, but it means that saved
  weights from previous versions will be incompatible. (`#41`_)

.. _#40: https://github.com/nengo/keras-lmu/pull/40
.. _#41: https://github.com/nengo/keras-lmu/pull/41

0.3.1 (November 16, 2020)
=========================

**Changed**

- Raise a validation error if ``hidden_to_memory`` or ``input_to_hidden`` are True
  when ``hidden_cell=None``. (`#26`_)

**Fixed**

- Fixed a bug with the autoswapping in ``keras_lmu.LMU`` during training. (`#28`_)
- Fixed a bug where dropout mask was not being reset properly in the hidden cell.
  (`#29`_)

.. _#26: https://github.com/nengo/keras-lmu/pull/26
.. _#28: https://github.com/nengo/keras-lmu/pull/28
.. _#29: https://github.com/nengo/keras-lmu/pull/29


0.3.0 (November 6, 2020)
========================

**Changed**

- Renamed module from ``lmu`` to ``keras_lmu`` (so it will now be imported via
  ``import keras_lmu``), renamed package from ``lmu`` to
  ``keras-lmu`` (so it will now be installed via ``pip install keras-lmu``), and
  changed any references to "NengoLMU" to "KerasLMU" (since this implementation is
  based in the Keras framework rather than Nengo). In the future the ``lmu`` namespace
  will be used as a meta-package to encapsulate LMU implementations in different
  frameworks. (`#24`_)

.. _#24: https://github.com/abr/lmu/pull/24

0.2.0 (November 2, 2020)
========================

**Added**

- Added documentation for package description, installation, usage, API, examples,
  and project information. (`#20`_)
- Added LMU FFT cell variant and auto-switching LMU class. (`#21`_)
- LMUs can now be used with any Keras RNN cell (e.g. LSTMs or GRUs) through the
  ``hidden_cell`` parameter. This can take an RNN cell (like
  ``tf.keras.layers.SimpleRNNCell`` or ``tf.keras.layers.LSTMCell``) or a feedforward
  layer (like ``tf.keras.layers.Dense``) or ``None`` (to create a memory-only LMU).
  The output of the LMU memory component will be fed to the ``hidden_cell``.
  (`#22`_)
- Added ``hidden_to_memory``, ``memory_to_memory``, and ``input_to_hidden`` parameters
  to ``LMUCell``, which can be used to enable/disable connections between components
  of the LMU. They default to disabled. (`#22`_)
- LMUs can now be used with multi-dimensional memory components. This is controlled
  through a new ``memory_d`` parameter of ``LMUCell``. (`#22`_)
- Added ``dropout`` parameter to ``LMUCell`` (which applies dropout to the input)
  and ``recurrent_dropout`` (which applies dropout to the ``memory_to_memory``
  connection, if it is enabled). Note that dropout can be added in the hidden
  component through the ``hidden_cell`` object. (`#22`_)

**Changed**

- Renamed ``lmu.lmu`` module to ``lmu.layers``. (`#22`_)
- Combined the ``*_encoders_initializer``parameters of ``LMUCell`` into a single
  ``kernel_initializer`` parameter. (`#22`_)
- Combined the ``*_kernel_initializer`` parameters of ``LMUCell`` into a single
  ``recurrent_kernel_initializer`` parameter. (`#22`_)

**Removed**

- Removed ``Legendre``, ``InputScaled``, ``LMUCellODE``, and ``LMUCellGating``
  classes. (`#22`_)
- Removed the ``method``, ``realizer``, and ``factory`` arguments from ``LMUCell``
  (they will take on the same default values as before, they just cannot be changed).
  (`#22`_)
- Removed the ``trainable_*`` arguments from ``LMUCell``. This functionality is
  largely redundant with the new functionality added for enabling/disabling internal
  LMU connections. These were primarily used previously for e.g. setting a connection to
  zero and then disabling learning, which can now be done more efficiently by
  disabling the connection entirely. (`#22`_)
- Removed the ``units`` and ``hidden_activation`` parameters of ``LMUCell`` (these are
  now specified directly in the ``hidden_cell``. (`#22`_)
- Removed the dependency on ``nengolib``. (`#22`_)
- Dropped support for Python 3.5, which reached its end of life in September 2020.
  (`#22`_)

.. _#20: https://github.com/abr/lmu/pull/20
.. _#21: https://github.com/abr/lmu/pull/21
.. _#22: https://github.com/abr/lmu/pull/22

0.1.0 (June 22, 2020)
=====================

Initial release of KerasLMU 0.1.0! Supports Python 3.5+.

The API is considered unstable; parts are likely to change in the future.

Thanks to all of the contributors for making this possible!
