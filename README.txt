You can source the galaxy-zoo dataset at
https://drive.google.com/file/d/1-W8lLL4_UCXsR66QAiKSNkGSWjmsr2rG/view?usp=drive_link

Unzip galaxy zoo file into FFT-GALAXY-CLASSIFICAITON directory(same folder as this README)

Correct File structure should look like

FFT-GALAXY-CLASSIFICAITON
    galaxy-zoo-the-galaxy-challenge
        all_ones_benchmark.zip
        all_zeros_benchmark.zip
        central_pixel_benchmark.zip
        images_test_rev1.zip
        images_training_rev1.zip
        training_solutions_rev1.zip
    galaxy_fft_model.py
    requirements.txt
    .gitignore
    README.txt

Setup (Windows)
Open this repository with VSCode
Assuming you have python installed(you should) run these commands below in the vscode terminal.

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m galaxy_fft_model

The program should run without error if the steps above are followed.