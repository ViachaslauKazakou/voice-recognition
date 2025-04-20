# Project settings


## 1. Create virtual environment using poetry

``` poetry install ```

If tou want create .venv inside project:

``` poetry config virtualenvs.in-project true ```

## 2. Instal ollama

## 3. Pull models

```ollama pull llama2```

``` ollama pull gemma ```

ollama pull codellama

ollama pull mixtral

- [How to start](how_to_start.md)

### VScode settings

```
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run app",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/main.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "cwd": "${workspaceFolder}",
            "args": [
                "-s",
                "-v",
                // "${file}"
                "${workspaceFolder}/tests/test_dice100.py"
            ],
            "console": "integratedTerminal",
            "purpose": ["debug-test"],
            "python": "${command:python.interpreterPath}"

        }
    ]
    ```
