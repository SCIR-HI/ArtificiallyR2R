# My Project

Welcome to this project. This README aims to guide you through the purpose and functionality of each `.py` file in the repository.

## Files and their Functions:

### 1. **data_handler.py**
- **Purpose**: Manages all data-related processes.
- **Features**:
    - Reading data from sources.
    - Pre-processing of data.
    - Defining the dataset structure.
    - `collate_fn` for custom batch creation.

### 2. **myparser.py**
- **Purpose**: Handles command-line arguments for training.
- **Features**:
    - Parsing training settings from the command line.

### 3. **trainer.py**
- **Purpose**: Defines the training logic.
- **Features**:
    - Trainer class definition.
    - Launch training using `trainer.run`.

### 4. **evaluator.py**
- **Purpose**: Code to compute different evaluation metrics.
- **Features**:
    - Provides metric calculation post-training/testing.

### 5. **run.py**
- **Purpose**: Main entry point to initialize and start the training.
- **Features**:
    - Initializes necessary modules and datasets.
    - Launches the training.

### 6. **test.py**
- **Purpose**: Similar to `run.py` but for launching testing.
- **Features**:
    - Sets up testing environment.
    - Runs the testing loop.

### 7. **utils.py**
- **Purpose**: Utility functions to support main processes.
- **Features**:
    - Model initialization.
    - Tokenizer setup.
    - Optimizer initialization.
    - Functions to record running time.
    - Load or save the model to/from disk.

## Getting Started:

To begin training, run:
    bash run.sh

For testing, use:
    bash test.sh