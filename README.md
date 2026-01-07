# Noise-Adaptive rPPG Framework via Rayleigh Quotient Optimization and Motion Artifacts Classification

This repository contains the main code for the paper **"Noise-Adaptive rPPG Framework via Rayleigh Quotient Optimization and Motion Artifacts Classification."** 

## Key Functionalities

1. **motion_class.ipynb**  
   Motion classification using Convolutional Neural Networks (CNNs). This script processes the STFT to classify motion artifacts, which are then used to improve the quality of rPPG signal extraction.

2. **adaptiveFilters.py**  
   A script for strong aperiodic noise cancellation using adaptive filters based on Affine Projection. It is used to filter out random noise from the rPPG signal.

3. **SpectralSubtraction.py**  
   A method for strong periodic noise cancellation using enhanced spectral subtraction. This helps in removing strong periodic noise from the signal.

4. **RQO.py**  
   Implementation of the Rayleigh Quotient Optimization (RQO) algorithm for weak noise suppression in the rPPG signal.

5. **HMM.py**  
   Implementation of heart rate trajectory tracking based on Hidden Markov Models (HMM). It enables robust heart rate estimation, even in the presence of sudden fluctuations.

### Required Libraries

- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib`
- `pytorch`
- `pyVHR`

## Usage

### Data Preprocessing

Run the  script in data_preprocess to prepare the dataset for further processing. This involve:

- RGB signal extraction from videos
- Motion signal extraction from videos
- Feature extraction for motion classification

### Motion Classification

Use the `motion_class.ipynb` notebook to classify different types of motion artifacts in the video frames, which are then used for noise adaptation.

### Heart Rate Estimation

Run the relevant scripts for rPPG signal extraction:

- Use `RQO.py` for weak noise suppression in the rPPG signal.
- Use `SpectralSubtraction.py` to remove periodic noise.
- Use `adaptiveFilters.py` to remove aperiodic noise.

Finally, estimate the heart rate using `HMM.py`, which tracks heart rate trajectories.

