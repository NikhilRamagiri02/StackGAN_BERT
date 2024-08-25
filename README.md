StackGAN-BERT/
│
├── data/
│   ├── CUB_200_2011/           # Bird dataset used for training
│   ├── embeddings/             # Folder containing BERT-generated text embeddings
│
├── models/                     # Implementation of StackGAN model
│
├── scripts/                    # Utility scripts for training, testing, and image generation
│   ├── bert_emb.py             # Script for generating BERT embeddings from text
│   ├── train.py                # Script for training the StackGAN model
│   ├── generate.py             # Script for generating images based on text input
│
├── config.py                   # Configuration file with model hyperparameters and paths
│
└── app/                        # Code for the web interface (React frontend, Flask/Streamlit backend)

