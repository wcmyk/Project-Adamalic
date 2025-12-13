Contributing
============

Please see the main `CONTRIBUTING.md <../CONTRIBUTING.md>`_ file for detailed contribution guidelines.

Quick Overview
--------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/Project-Adamalic.git
   cd Project-Adamalic

   # Install in development mode
   pip install -e ".[dev]"

   # Run tests
   pytest

Code Style
----------

We follow PEP 8 with these tools:

.. code-block:: bash

   # Format code
   black .

   # Sort imports
   isort .

   # Type checking
   mypy LILITH/ SHAMSHEL/

   # Linting
   flake8 LILITH/ SHAMSHEL/

Testing
-------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=LILITH --cov=SHAMSHEL

   # Run specific test file
   pytest tests/test_lilith/test_model.py
