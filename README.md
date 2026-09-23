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
```bash
human-behavior/
│
├── .github/                  # CI/CD workflows, issue templates
│   └── workflows/
│
├── configs/                  # Config for modules, model,...
│
├── data/               
│   ├── raw/                 
│   ├── processed/           
│   └── annotations/          
│
├── docs/                   
│
├── models/                  
│
├── notebooks/              
│
├── src/                      # Main source code 
│   ├── __init__.py
│   │
│   ├── core/                 # Core frequently using (ROIs, CV2 libs,...)
│   │
│   ├── modules/              # Specific modules for each behavior issues
│   │   ├── __init__.py
│   │   ├── exam_cheating/    # Checking student cheating or no
│   │   ├── hand_washing/     # Checking washing stage
│   │   └── staff_absence/    # Spectate employee at workstation
│   │
│   └── utils/                # Shared utils
│
├── tests/                    # Unit tests 
├── .gitignore                
├── Dockerfile                
├── LICENSE                   
├── README.md                 
└── requirements.txt          
```