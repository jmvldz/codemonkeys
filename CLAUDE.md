# CODEMONKEYS

## Setup & Installation
```bash
# Install dependencies
pip install -e .
```

## Environment Configuration
```bash
export CODEMONKEYS_PRIMARY_API_KEY="your_key_here"
export CODEMONKEYS_PRIMARY_MODEL_NAME="claude-3-5-sonnet-20241022"
export CODEMONKEYS_PRIMARY_MODEL_TYPE="anthropic"
```

## Commands
```bash
# Run the main program
python codemonkeys/run_codemonkeys.py trajectory_store_dir=$TRAJ_STORE_DIR

# Type checking
pyright

# Run specific test (example)
python -m pytest tests/path/to/test.py::test_function -v
```

## Code Style
- Type hints: Required with generics where appropriate
- Imports: Group by package, standard library first
- Dataclasses: Use for structured data
- Error handling: Assertions for validation
- Naming: snake_case for functions/variables, PascalCase for classes
- File structure: Modular with clear separation of concerns
- Documentation: Docstrings for functions and classes