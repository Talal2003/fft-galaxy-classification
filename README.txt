Galaxy Zoo FFT Model Comparison

A comparison of FFT-based galaxy classification models on the
Galaxy Zoo training dataset. Multiple model implementations run against the
same feature matrix so results are directly comparable.

Dataset source:
https://drive.google.com/file/d/1-W8lLL4_UCXsR66QAiKSNkGSWjmsr2rG/view?usp=drive_link

Unzip the dataset into the repository root so the layout looks like this:

fft-galaxy-classification
    galaxy-zoo-the-galaxy-challenge
        all_ones_benchmark.zip
        all_zeros_benchmark.zip
        central_pixel_benchmark.zip
        images_test_rev1.zip
        images_training_rev1.zip
        training_solutions_rev1.zip
    fft_galaxy_runner
    docs
    reports
    galaxy_fft_model.py
    config.toml
    requirements.txt
    README.txt

Setup (Windows)

Clone the repository into a folder with GitHub Desktop.
Open the folder in VS Code.
Run these commands in the VS Code terminal:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python galaxy_fft_model.py

The program should run without error if the steps above are followed.

Setup (macOS / Linux)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python galaxy_fft_model.py

Usage

All settings are in config.toml. Edit that file, then run:

    python galaxy_fft_model.py

To use a different config file, pass its path as the first argument:

    python galaxy_fft_model.py path/to/other-config.toml

Output is printed to the terminal. To also write a report file, set
output_path in config.toml:

    output_path = "output/test.txt"

Config fields in config.toml

    sample_size    - number of images to load, or "all" for the full dataset
    targets        - "all" (37 outputs) or "q1" (3 Question 1 outputs)
    evaluation     - "single" (single split) or "twofold" (even/odd folds)
    validation_split - fraction held out for testing, e.g. 0.2
    random_seed    - for reproducible splits
    color_mode     - "BW" or "rgb" (rgb extracts FFT per channel)
    models         - list of models to run, or ["all"]
    detailed       - true to print per-target RMSE breakdown
    output_path    - optional path for a plain-text report file
    image_size     - input image resize dimensions, e.g. [128, 128]
    fft_size       - FFT spectrum crop size, e.g. [32, 32]

Available models

    sklearn-linear     - scikit-learn LinearRegression (from main branch)
    closed-form-linear - normal equation solver (from 2-fold-cross branch)
    ridge              - Ridge regression
    pca-linear         - PCA dimensionality reduction + LinearRegression
    random-forest      - RandomForestRegressor
