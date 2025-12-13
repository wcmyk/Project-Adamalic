Examples
========

This page provides examples of using Project Adamalic components.

Training Examples
-----------------

Simple Model Training
~~~~~~~~~~~~~~~~~~~~~

See ``examples/train_simple_model.py`` for a basic training example.

.. literalinclude:: ../examples/train_simple_model.py
   :language: python
   :caption: Simple model training

Advanced Training
~~~~~~~~~~~~~~~~~

See ``examples/advanced_training.py`` for advanced features:

* Mixed precision training
* Gradient accumulation
* Early stopping
* Validation sets
* Regular checkpointing

Sampling Examples
-----------------

Advanced Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/advanced_sampling.py`` for different sampling methods:

* Greedy sampling
* Temperature sampling
* Top-k sampling
* Top-p (nucleus) sampling
* Beam search

LoRA Fine-tuning
----------------

Efficient Fine-tuning with LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/lora_finetuning.py`` for parameter-efficient fine-tuning:

* Applying LoRA to models
* Training only adapter parameters
* Saving/loading LoRA checkpoints

Sandbox Examples
----------------

Secure Code Execution
~~~~~~~~~~~~~~~~~~~~~

See ``examples/shamshel_sandbox.py`` for sandbox usage:

* Running safe code
* Security validation
* Timeout enforcement
* Automatic cleanup

All Examples
------------

All example scripts are available in the ``examples/`` directory:

* ``train_simple_model.py`` - Basic model training
* ``advanced_training.py`` - Advanced training features
* ``advanced_sampling.py`` - Different sampling strategies
* ``lora_finetuning.py`` - LoRA fine-tuning
* ``shamshel_sandbox.py`` - Sandbox execution
