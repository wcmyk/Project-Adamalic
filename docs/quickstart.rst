Quickstart Guide
================

This guide will help you get started with Project Adamalic quickly.

Installation
------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/wcmyk/Project-Adamalic.git
   cd Project-Adamalic

2. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Basic Usage
-----------

Training a Simple Model (LILITH)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LILITH import ModelConfig, TrainingConfig, train

   # Sample corpus
   corpus = ["Hello world", "Machine learning is great"]

   # Configure model and training
   model_config = ModelConfig(vocab_size=100, d_model=128)
   train_config = TrainingConfig(max_steps=1000, batch_size=4)

   # Train
   model = train(corpus, model_config, train_config)

Using the Sandbox (SHAMSHEL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from SHAMSHEL import SandboxRunner

   runner = SandboxRunner(timeout_sec=5, max_memory_mb=100)

   code = "print('Hello from sandbox!')"
   result = runner.run_python(code)

   print(result.stdout)  # "Hello from sandbox!"

Advanced Features
-----------------

Advanced Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LILITH import sample_with_strategy

   # Different sampling strategies
   greedy = sample_with_strategy(model, prompt, strategy="greedy")
   nucleus = sample_with_strategy(model, prompt, strategy="top_p", top_p=0.9)
   beam = sample_with_strategy(model, prompt, strategy="beam", beam_width=4)

LoRA Fine-tuning
~~~~~~~~~~~~~~~~

.. code-block:: python

   from LILITH import apply_lora_to_model, get_lora_parameters

   # Apply LoRA to model
   lora_model = apply_lora_to_model(
       base_model,
       target_modules=["head"],
       rank=4,
       alpha=1.0
   )

   # Only LoRA parameters are trainable
   lora_params = get_lora_parameters(lora_model)

Next Steps
----------

* Check out the :doc:`examples` for more usage patterns
* Read the :doc:`api/index` for detailed API documentation
* See :doc:`contributing` to contribute to the project
