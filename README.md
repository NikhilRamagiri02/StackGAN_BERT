StackGAN-BERT: Text-to-Image Generation using StackGAN and BERT
Overview

This project implements a text-to-image generation system that combines the power of Stacked Generative Adversarial Networks (StackGAN) with BERT embeddings. The system takes a text description as input and generates high-resolution images that match the description. This implementation is particularly focused on generating images of birds, leveraging the CUB-200-2011 dataset for training and evaluation.
Features

    Text-to-Image Generation: Generate photo-realistic images based on textual descriptions using a two-stage StackGAN architecture.
    BERT Embeddings: Utilize BERT embeddings to capture the contextual meaning of input text, leading to more accurate image generation.
    High-Resolution Images: The system generates images in two stages, with the second stage refining the image to a high resolution.
    Customizable Inputs: Users can input detailed descriptions to customize the generated images.

Project Structure

    data/: Contains datasets used for training and testing.
        CUB_200_2011/: Bird dataset used for training.
        embeddings/: Folder containing BERT-generated text embeddings.
    models/: Contains the implementation of the StackGAN model, including the generator, discriminator, and conditional augmentation modules.
    scripts/: Utility scripts for training, testing, and image generation.
    bert_emb.py: Script for generating BERT embeddings from textual descriptions.
    train.py: Script for training the StackGAN model.
    generate.py: Script for generating images based on text inputs.
    config.py: Configuration file containing model hyperparameters and paths.
    app/: Contains the code for the web interface (React frontend and Flask/Streamlit backend).

Installation

    Clone the Repository:

    bash

git clone https://github.com/yourusername/StackGAN-BERT.git
cd StackGAN-BERT

Set Up the Environment:

    Create a virtual environment using Anaconda:

    bash

conda create --name stackgan-bert python=3.9
conda activate stackgan-bert

Install the required dependencies:

bash

    pip install -r requirements.txt

Download the Dataset:

    Download the CUB-200-2011 dataset and place it in the data/ folder.

Generate BERT Embeddings:

    Run the script to generate BERT embeddings for the dataset:

    bash

        python scripts/bert_emb.py

Usage
Training the Model

To train the StackGAN model with BERT embeddings:

bash

python train.py --config config.py

Generating Images

To generate images based on custom text input:

bash

python generate.py --text "A small bird with blue feathers."

Web Interface

To run the web interface for interactive text-to-image generation:

    Run the Backend:
        If using Flask:

        bash

cd app/server
python app.py

If using Streamlit:

bash

    streamlit run app.py

Run the React Frontend:

bash

cd app/client
npm install
npm start