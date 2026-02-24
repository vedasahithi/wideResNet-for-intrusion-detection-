# TWSA: Taylor Wave Search Optimization with WideResNet for Intrusion Detection and Mitigation in Cloud Computing

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 |
| RAM | 8 GB |
| Storage | 100+ GB |
| GPU | Required |
| CPU | 1.7 GHz or higher |

## Software Requirements

### Python 3.9.11
- Download: https://www.python.org/downloads/release/python-396/
- Installer: **Windows x86-64 executable installer**
- Refer to `steps to install python.doc` for detailed installation instructions.

### PyCharm 2020.3.3
- Download: https://www.jetbrains.com/pycharm/download/other.html

## How to Run

### Step 1: Load the Project in PyCharm

1. Open PyCharm.
2. Go to **File → Open**, browse to the project folder, and select it.
3. Wait for PyCharm to finish loading settings (progress shown at the bottom of the screen).
4. Verify the Project Interpreter: **File → Settings → Project: 294417 → Project Interpreter**
   - The path should point to:  
     `C:\Users\---\AppData\Local\Programs\Python\Python39-64\python.exe`
   - If not present, add the `python.exe` from your installed Python location.
5. In the PyCharm Terminal (bottom left), run:
   ```
   pip install -r requirements.txt
   ```

### Step 2: Run the Program

1. In the project panel, open **294417 → Main → GUI.py** and click the **Run** button.
2. In the GUI window:
   - Enter a feature dimension (e.g., 5, 10, 15, 20)
   - Click **START** — results will be displayed after processing.
   - Click **Run Graph** to view the result graph.
3. ⏱️ **Expected execution time: 15–20 minutes**

### Step 3: Generate Paper Graphs

1. Open **294417 → Main → Result_graphs.py** and click the **Run** button.

## Project File Descriptions

| File | Description |
|------|-------------|
| `Main/GUI.py` | User interface — entry point of the application |
| `Main/Run.py` | Main execution code |
| `Main/Data_Normalization.py` | Data normalization using Min-Max normalization |
| `Main/Feature_selection.py`, `HSBOA.py` | Feature selection using SVM-RFE with HSBOA |
| `Proposed_TWSA_WRN/WideResnet.py`, `TWSA.py` | Intrusion detection and attack mitigation using the proposed TWSA-WRNet |
