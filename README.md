# Human Behavior Modules
All modules using Computer Vision models (YOLO) for human behavior

# Preparation
## Install venv + active venv + dependencies (current version: Python 3.10.11)
Create venv active
```bash
py -m venv .venv
```
Active venv
```bash
.venv\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
### Download model (onnx)
```bash
py models\download_model.py
```

# Repo Structure
Read at docs/architecture.md