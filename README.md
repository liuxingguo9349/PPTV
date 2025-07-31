# PPTV: Official Implementation for "Explore the Ideology of Deep Learning in ENSO Forecasts"

This repository contains the official code implementation for the paper: **Explore the Ideology of Deep Learning in ENSO Forecasts**.

We propose a mathematically grounded interpretability method, Practical Partial Total Variation (PPTV), to break the black box of deep learning models for ENSO forecasting. This code allows you to reproduce the main findings, including the PPTV heatmaps, comparisons with other methods, and the retraining validation experiments presented in the paper.

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/liuxingguo9349/PPTV.git
cd PPTV
```

### 2. Create Conda Environment
The code was originally developed using TensorFlow 1.x. We recommend using Conda to create a consistent environment. You can create an `environment.yml` file with the content below and run `conda env create -f environment.yml`.

**`environment.yml` content:**
```yml
name: pptv-enso-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - tensorflow-gpu=1.15
  - numpy
  - matplotlib
  - netcdf4
  - xarray
  - scipy
  - tqdm
```
Then activate the environment:
```bash
conda activate pptv-enso-env
```

### 3. Download Data
The models are trained and evaluated on the **GODAS**, **CMIP5**, and **SODA** datasets. Please download them from their official sources and place them in a `data/` directory.

*   **GODAS:** [psl.noaa.gov/data/gridded/data.godas.html](https://www.psl.noaa.gov/data/gridded/data.godas.html)
*   **CMIP5:** [esgf-node.llnl.gov/projects/cmip5/](https://esgf-node.llnl.gov/projects/cmip5/)
*   **SODA:** [climatedataguide.ucar.edu/climate-data/soda-simple-ocean-data-assimilation](https://climatedataguide.ucar.edu/climate-data/soda-simple-ocean-data-assimilation)

### 4. Pre-trained Models
Place them in a `pretrained_models/` directory.

## How to Reproduce Results

This repository contains a direct copy of the development scripts. Each script is independent. The general workflow to reproduce the paper's findings is as follows:

### Step 1: Generate Importance Maps (Heatmaps)

The core logic for generating heatmaps is located in the `src/` directory. You will need to run these scripts individually.

*   **For PPTV:** Run the scripts with `grad` in their names.
    *   Example script: `src/grad_grad.cnn.sample.py`
*   **For Comparison Methods:** Run the scripts corresponding to each method.
    *   **Grad-CAM:** `src/gradcam_...py`
    *   **Score-CAM:** `src/scorecam_...py`

**Note:** You may need to open these scripts and manually adjust the paths to the pre-trained models and input data to match your local setup.

### Step 2: Run Retraining Validation

The scripts for performing the retraining validation experiments (Fig. 1b) are located in the `scripts/` directory.

*   Look for scripts with `retrain` in their names, such as:
    *   `scripts/retrainmul&concat_...py`
    *   `scripts/retrainscorecam_...py`
*   These scripts typically take the heatmaps generated in Step 1 as input (to be used as masks). You will likely need to edit the script to provide the correct path to these masks.

### Step 3: Plot Figures

The scripts used to generate the final figures for the paper (Fig. 1-5) are located in the `scripts/` directory.

*   Look for scripts with `figure` in their names, for example:
    *   `scripts/figure_figure_PTPV.py` (to plot PPTV results)
    *   `scripts/figure_figure_springforecast.py` (to plot seasonal analysis)
    *   `scripts/figure_relationcnn.py` (to plot correlation skills)
*   Run these scripts after you have generated all the necessary data from the steps above. The scripts will read the data from your results directories and create the plots.
