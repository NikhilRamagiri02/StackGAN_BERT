import streamlit as st
import time
import torch
from transformers import BertTokenizer, BertModel
from layers import Stage2Generator, Stage1Generator
import matplotlib.pyplot as plt
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to display the image
def display_image(image_tensor):
    image_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy().squeeze()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    plt.imshow(image_np)
    plt.axis('off')
    st.pyplot()

st.title('AI Image Generator')


# Load the trained models and tokenizer
Stage1_G = Stage1Generator(emb_dim=768)
Stage1_G.load_state_dict(torch.load("E:\\StackGAN-BERT\\output\\model\\netG_epoch_600.pth"))
Stage1_G.eval()

generator = Stage2Generator(Stage1_G, emb_dim=768)
generator.load_state_dict(torch.load("E:\\StackGAN-BERT\\output2\\model\\netG_epoch_20.pth"))
generator.eval()

tokenizer = BertTokenizer.from_pretrained("E:\\StackGAN-BERT\\input\\data\\bert_base_uncased")
model = BertModel.from_pretrained("E:\\StackGAN-BERT\\input\\data\\bert_base_uncased")
model.eval()

# Get user input

text = st.text_input('', '')


# Process the textual description
tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
text_embedding = model(**tokenized_text)["last_hidden_state"].mean(dim=1)

# Generate a noise vector
noise_vector = torch.rand(1, 100)

# Display the input and generate image
if st.button('Generate Image'):
    with torch.no_grad():
        generated_image = generator(text_embedding, noise_vector)
    display_image(generated_image)
  
