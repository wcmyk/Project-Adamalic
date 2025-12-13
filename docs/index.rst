Project Adamalic (AngelOS) Documentation
========================================

Welcome to the documentation for Project Adamalic, a distributed multi-agent AI operating system.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api/index
   examples
   contributing

Overview
--------

Project Adamalic (AngelOS) is an ambitious distributed multi-agent AI operating system with specialized components:

* **LILITH**: Dual-role LLM system with general and code-focused models
* **SHAMSHEL**: Secure sandbox runner for code execution
* **ADAM**: Supervisor angel (planned)
* **Additional angels**: Various specialized components (planned)

Features
--------

LILITH Features
~~~~~~~~~~~~~~~

* GPT-style transformer decoder
* Character-level and BPE tokenization
* Advanced sampling strategies (top-k, top-p, beam search)
* LoRA fine-tuning support
* Mixed precision training
* KV-cache for efficient inference
* Comprehensive evaluation metrics

SHAMSHEL Features
~~~~~~~~~~~~~~~~~

* Secure code execution with resource limits
* AST-based security validation
* Automatic cleanup of temporary directories
* Infinite loop detection
* Memory and CPU limits
* Support for Python and shell scripts

Quick Links
-----------

* :doc:`quickstart` - Get started quickly
* :doc:`api/index` - API Reference
* :doc:`examples` - Example usage
* :doc:`contributing` - Contributing guidelines

Installation
------------

.. code-block:: bash

   pip install -e .

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
