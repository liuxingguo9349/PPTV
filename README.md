# PPTV: Official Implementation for "Explore the Ideology of Deep Learning in ENSO Forecasts"

We propose a mathematically grounded interpretability method, Practical Partial Total Variation (PPTV), to break the black box of deep learning models for ENSO forecasting. PPTV helps identify critical geographic regions for prediction and analyzes the model's behavior under different conditions, such as the Spring Predictability Barrier (SPB).

## Repository Structure

```
.
├── data/                 # Data downloading and preprocessing instructions
├── src/                  # Core source code
│   ├── models.py         # CNN model architecture
│   ├── pptv.py           # PPTV method implementation
│   └── comparison_methods.py # Implementations for Grad-CAM, etc.
├── scripts/              # Scripts to run experiments and generate figures
│   ├── 1_run_interpretability.py
│   ├── 2_run_retrain_validation.py
│   └── 3_plot_figures.py
├── results/              # Directory to save generated figures and data (ignored by git)
├── environment.yml       # Conda environment file
└── README.md
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/liuxingguo9349/PPTV.git
cd PPTV
```

### 2. Create Conda Environment
We recommend using Conda to manage the dependencies.
```bash
conda env create -f environment.yml
conda activate pptv-env
```

### 3. Download Data
The models are trained and evaluated on the GODAS, CMIP5, and SODA datasets. Please follow the instructions in `data/README.md` to download and place the data in the correct directories.

### 4. Pre-trained Models
Place them in a `pretrained_models/` directory.

## Usage: Reproducing Paper Results

The following scripts allow you to reproduce the key findings of our paper.

### 1. Run PPTV and other Interpretability Methods
To generate the importance maps for a given lead time (e.g., 1 month), run:
```bash
python scripts/1_run_interpretability.py --method pptv --lead_time 1 --output_dir results/maps
```
*   `--method`: Can be `pptv`, `gradcam`, `perturbation`, `vbp`.
*   This will save the generated `.nc` or `.npy` files in the specified output directory.

### 2. Run Retraining Validation
To validate the importance of the regions identified by PPTV (as in Fig. 1b), run:
```bash
python scripts/2_run_retrain_validation.py --mask_path results/maps/pptv_lead1.nc --output_dir results/retrain_skill
```

### 3. Generate Figures
To generate the figures from the paper (e.g., Fig. 1, 2, 3, 4, 5) using the pre-computed results, run:
```bash
python scripts/3_plot_figures.py --results_path results/ --figure_id 1
```
*   `--figure_id`: Specifies which figure to plot (e.g., 1, 2, 3, 4, 5, or 'all').
