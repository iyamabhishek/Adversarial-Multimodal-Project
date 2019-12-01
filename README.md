# Adversarial-Multimodal-Project
This repository contains the code for the adversarial experiments in a multimodal setting. We use the Memory Fusion Network on CMU-MOSI dataset for our experiments.

The data folder contains the data for trimodal, bimodal and unimodal experiments.

To run bimodal and unimodal experiments specify, the modality using t, a, and v for text, audio, and video respectively in the load_saved_data method. Accordingly please update the config["input_dims"] variable to indicate the dimension of each modality. The order of inputs_dims depends on the order of the input modality as specified in the load_saved_data method.

Text dimension = 300
Audio dimension = 5
Video dimension = 20
