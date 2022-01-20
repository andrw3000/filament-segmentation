# Filament Segmentation
A two-step instance segmentation of filaments in electron microspic tomograms.

- We provide both U-Net and Autoencoder architectures in `models`.
- In `utils` varous conversion files are included to process MRC file type data and give read-outs into STAR files. Intermediate images and demonstrations are given as PNG images.
- The `demos` directory contains python notebooks giving examples of how to train and predict with the model.
- Clone this repo `filament-segmentation` and add data into a new directory called `data`.
