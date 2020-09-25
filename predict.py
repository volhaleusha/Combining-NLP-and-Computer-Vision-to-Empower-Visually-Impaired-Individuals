import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import Encoder, Decoder, EncoderAtt, DecoderAtt, Transformer
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import json
import torch.nn.functional as F
from utils import load_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    if args.model_type == 'no_attention':
        encoder = Encoder(args.embed_size).to(device)
        decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    elif args.model_type == 'attention':
        encoder = EncoderAtt(encoded_image_size=9).to(device)
        decoder = DecoderAtt(vocab, args.encoder_dim,  args.hidden_size, args.attention_dim,  
                 args.embed_size, args.dropout_ratio, args.alpha_c).to(device)
        
    elif args.model_type == 'transformer':   
        
        model = Transformer(len(vocab), args.embed_size, args.transformer_layers , 8, args.dropout_ratio).eval()    
        
    else:
        print('Select model_type attention or no_attention')
      
    if args.model_type != 'transformer':
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(args.encoder_path ,map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(args.decoder_path ,map_location=torch.device('cpu')))
    else:
        model = model.to(device)
        model.load_state_dict(torch.load(args.model_path ,map_location=torch.device('cpu')))    
    
 
    filenames = os.listdir(args.image_dir)  
    
    predicted ={}
    
    for file in tqdm(filenames):
        if file  == '.DS_Store':
            continue
        # Prepare an image
        image = load_image(os.path.join(args.image_dir,file), transform)
        image_tensor = image.to(device)
        
        if args.model_type == 'attention':
            features = encoder(image_tensor)
            sampled_ids, _ = decoder.sample(features)
            sampled_ids = sampled_ids[0].cpu().numpy()
            sampled_caption = ['<start>']
        elif args.model_type == 'no_attention':
            features = encoder(image_tensor)
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids[0].cpu().numpy()
            sampled_caption = ['<start>']
            
        elif args.model_type == 'transformer':
            e_outputs = model.encoder(image_tensor)
            max_seq_length = 20 
            sampled_ids = torch.zeros(max_seq_length, dtype = torch.long)
            sampled_ids[0] = torch.LongTensor([[vocab.word2idx['<start>']]]).to(device)
        
            for i in range(1, max_seq_length):    
            
                trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
                trg_mask= Variable(torch.from_numpy(trg_mask) == 0).to(device)
 
                out = model.decoder(sampled_ids[:i].unsqueeze(0), e_outputs, trg_mask)

                out = model.out(out)
                out = F.softmax(out, dim=-1)
                val, ix = out[:, -1].data.topk(1)
                sampled_ids[i] = ix[0][0]
                    
            sampled_ids = sampled_ids.cpu().numpy()
            sampled_caption = []

        # Convert word_ids to words
        #sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        #print(sentence)
        predicted[file] = sentence
        #print(file, sentence)
        
    json.dump(predicted, open(args.predict_json, 'w'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/val2017', help='directory for resized images')
    parser.add_argument('--encoder_path', type=str, default='models_tr/encoder-att-8.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-att-8.ckpt', help='path for trained decoder')
    parser.add_argument('--model_path', type=str, default='models_tr_3/model_tr3_21.ckpt' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--predict_json', default='output/predict_val_coco_tr.json', help='output json file')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--model_type', type=str , default='transformer', help='no_attention or attention')
    parser.add_argument('--encoder_dim', type=int, default= 2048, help = 'dimension of image embeddings image embeddings in attention model')
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--alpha_c', type=float, default= 1.0, help = 'coefficient to calculate attention loss')
    parser.add_argument('--fine_tune',  default= True, help = 'freeze lower layers of resnet')
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--transformer_layers', type=int , default= 6, help='number of transformer layers')
    
    
    
    args = parser.parse_args()
    main(args)
