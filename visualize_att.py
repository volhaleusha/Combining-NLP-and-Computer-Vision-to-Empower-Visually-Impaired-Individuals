import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import os
import pickle
from build_vocab import Vocabulary
from model import Encoder, Decoder, EncoderAtt, DecoderAtt
from utils import load_image


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

        
    encoder = EncoderAtt(encoded_image_size= 9).eval()
    decoder = DecoderAtt(vocab, args.encoder_dim,  args.hidden_size, args.attention_dim,  
                 args.embed_size, args.dropout_ratio, args.alpha_c).eval()

        
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))

    # Prepare an image
    image = load_image(os.path.join(args.image_path, args.image), transform)
    image_tensor = image.to(device)
    sampled_caption =[]
    
    features = encoder(image_tensor)
    sampled_ids, alphas  = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    words = ['<start>']
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        words.append(word)
        if word == '<end>':
            break
    print(words)

    plt.figure(figsize=(15, 4))
    for t in range(len(words)):
        
        
        plt.subplot(np.ceil(len(words) / 8.) , 8, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(np.asarray(Image.open(os.path.join(args.image_path, args.image)).resize([224, 224], Image.LANCZOS)))
        current_alpha = alphas[t].view(-1, 9)
        alpha = skimage.transform.pyramid_expand(current_alpha.detach().numpy(), upscale=24, sigma=8)

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, default='VizWiz_val_00000005.jpg', help='input image for generating caption')
    parser.add_argument('--image_path', type=str, default = 'data/val/', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models2/encoder-att-10.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models2/decoder-att-10.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_epochs', type=int, default=10, help = 'number of epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--encoder_dim', type=int, default=2048, help = 'dimension of image embeddings image embeddings in attention model')
    parser.add_argument('--alpha_c', type=float, default= 1.0, help = 'coefficient to calculate attention loss')

    args = parser.parse_args()


    # Visualize caption and attention of best sequence
    main(args)
