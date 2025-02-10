#!/bin/bash
# run_tests.sh

# Asegura que PYTHONPATH incluya el directorio actual
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ejecuta los tests
pytest tests/ -v --cov=vyper --cov-report=term-missing