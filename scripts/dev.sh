#!/bin/bash

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Función para imprimir mensajes con color
print_message() {
    echo -e "${2}${1}${NC}"
}

# Función para verificar si el comando anterior fue exitoso
check_error() {
    if [ $? -ne 0 ]; then
        print_message "Error: $1" "${RED}"
        exit 1
    fi
}

# Función para instalar dependencias
setup() {
    print_message "Installing dependencies..." "${YELLOW}"
    poetry install
    check_error "Failed to install dependencies"
    
    print_message "Setting up pre-commit hooks..." "${YELLOW}"
    poetry run pre-commit install
    check_error "Failed to setup pre-commit hooks"
    
    print_message "Setup completed successfully!" "${GREEN}"
}

# Función para ejecutar tests
run_tests() {
    print_message "Running tests..." "${YELLOW}"
    if [ "$1" == "--coverage" ]; then
        poetry run pytest --cov=vyper --cov-report=html
    else
        poetry run pytest
    fi
    check_error "Tests failed"
}

# Función para ejecutar linters
lint() {
    print_message "Running black..." "${YELLOW}"
    poetry run black .
    check_error "Black formatting failed"
    
    print_message "Running isort..." "${YELLOW}"
    poetry run isort .
    check_error "Import sorting failed"
    
    print_message "Running mypy..." "${YELLOW}"
    poetry run mypy .
    check_error "Type checking failed"
    
    print_message "Running flake8..." "${YELLOW}"
    poetry run flake8 .
    check_error "Linting failed"
    
    print_message "All linting passed!" "${GREEN}"
}

# Función para ejecutar la aplicación en modo desarrollo
dev() {
    print_message "Starting development server..." "${YELLOW}"
    poetry run python main.py
}

# Función para ejecutar los contenedores de desarrollo
docker_dev() {
    print_message "Starting development containers..." "${YELLOW}"
    docker-compose up --build
}

# Función para limpiar archivos temporales
clean() {
    print_message "Cleaning temporary files..." "${YELLOW}"
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type d -name ".pytest_cache" -exec rm -r {} +
    find . -type d -name ".mypy_cache" -exec rm -r {} +
    find . -type d -name ".coverage" -exec rm -r {} +
    find . -type d -name "htmlcov" -exec rm -r {} +
    rm -f .coverage
    rm -f coverage.xml
    print_message "Cleanup completed!" "${GREEN}"
}

# Mostrar ayuda si no se proporcionan argumentos
if [ $# -eq 0 ]; then
    print_message "Usage: ./dev.sh [command]" "${YELLOW}"
    print_message "\nCommands:" "${YELLOW}"
    print_message "  setup         - Install dependencies and setup dev environment"
    print_message "  test         - Run tests"
    print_message "  test-cov     - Run tests with coverage"
    print_message "  lint         - Run all linters"
    print_message "  dev          - Start development server"
    print_message "  docker-dev   - Start development containers"
    print_message "  clean        - Clean temporary files"
    exit 0
fi

# Procesar comandos
case "$1" in
    "setup")
        setup
        ;;
    "test")
        run_tests
        ;;
    "test-cov")
        run_tests --coverage
        ;;
    "lint")
        lint
        ;;
    "dev")
        dev
        ;;
    "docker-dev")
        docker_dev
        ;;
    "clean")
        clean
        ;;
    *)
        print_message "Unknown command: $1" "${RED}"
        exit 1
        ;;
esac