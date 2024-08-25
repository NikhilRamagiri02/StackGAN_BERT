import torch
from transformers import BertTokenizer, BertModel
from layers import Stage2Generator, CAug
from layers import Stage1Generator, Stage1Discriminator
import matplotlib.pyplot as plt

def display_image(image_tensor):
   
    image_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy().squeeze()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()


Stage1_G = Stage1Generator(emb_dim=768)
Stage1_G.load_state_dict(torch.load("E:\\StackGAN-BERT\\output\\model\\netG_epoch_600.pth"))
Stage1_G.eval()


generator = Stage2Generator(Stage1_G, emb_dim=768)
generator.load_state_dict(torch.load("E:\\StackGAN-BERT\\output2\\model\\netG_epoch_50.pth"))
generator.eval()


tokenizer = BertTokenizer.from_pretrained("E:\\StackGAN-BERT\\input\\data\\bert_base_uncased")


model = BertModel.from_pretrained("E:\\StackGAN-BERT\\input\\data\\bert_base_uncased")
model.eval()

text = "A small yellow bird with a black crown and a short black pointed beak"
tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

text_embedding = model(**tokenized_text)["last_hidden_state"].mean(dim=1)


noise_vector = torch.randn(1, 100) 


with torch.no_grad():
    generated_image = generator(text_embedding, noise_vector)
parser = argparse.ArgumentParser(description='Process a sentence.')

# Add argument for the sentence
parser.add_argument('sentence', metavar='sentence', type=str, nargs='+',
                    help='a sentence to process')

# Parse the arguments
args = parser.parse_args()

# Combine the sentence into a single string
sentence = ' '.join(args.sentence)

prompt = sentence
seed = random.randint(0, sys.maxsize)

images = pipe(
    prompt = prompt,
    guidance_scale = 0.0,
    num_inference_steps = num_inference_steps,
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

low_res_image = images[0].resize((256, 256))
low_res_array = np.array(low_res_image)

# Add random noise to the low-resolution image
noise_factor = 0.5  # Adjust the noise factor as needed
noisy_low_res_array = low_res_array + noise_factor * np.random.randn(*low_res_array.shape)

# Clip the pixel values to be in the valid range [0, 255]
noisy_low_res_array = np.clip(noisy_low_res_array, 0, 255)

# Convert the noisy NumPy array back to a PIL image
noisy_low_res_image = Image.fromarray(noisy_low_res_array.astype(np.uint8))

print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
media.show_images([noisy_low_res_image])
noisy_low_res_image.save("output.jpg")
output_image = Image.open("output.jpg")

display_image(output_image)
