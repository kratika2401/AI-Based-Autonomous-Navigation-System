# Installation & Setup Guide

This guide describes how to set up the development environment for the 2D simulated autonomous navigation system.

---

## 1. Prerequisites
- **Python:** 3.10 or higher.
- **Operating System:** Platform independent (Windows, macOS, Linux).
- **Package Manager:** `pip` (standard with Python) or `virtualenv`.

---

## 2. Virtual Environment Setup

It is strongly recommended to use an isolated Python virtual environment.

### On Windows (PowerShell):
```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### On macOS / Linux:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

---

## 3. Dependency Installation

Install the project in editable mode along with development dependencies:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install in editable mode with development dependencies (pytest)
pip install -e ".[dev]"
```

To install runtime dependencies only:
```bash
pip install -e .
```

---

## 4. Verifying Installation

Verify that the dependencies are installed and the package can be resolved:
```bash
python -c "import autonomous_nav; import numpy; import pygame; print('Environment verified successfully')"
```

Run test suite:
```bash
pytest
```
