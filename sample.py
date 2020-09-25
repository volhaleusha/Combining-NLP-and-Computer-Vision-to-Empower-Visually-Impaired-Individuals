import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import Encoder, Decoder, EncoderAtt, DecoderAtt, Transformer
from PIL import Image
from torch.autograd import Variable
import json
from evaluate import bleu, cider, meteor, rouge, spice
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
        
        encoder = Encoder(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval()
    elif args.model_type == 'attention':   
        
        encoder = EncoderAtt(encoded_image_size= 9).eval()
        decoder = DecoderAtt(vocab, args.encoder_dim,  args.hidden_size, args.attention_dim,  
                 args.embed_size, args.dropout_ratio, args.alpha_c).eval()
        
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

    # Prepare an image
    image = load_image(os.path.join(args.image_path, args.image), transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    
    if args.model_type =='no_attention':
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
                 
    elif args.model_type =='attention':
        feature = encoder(image_tensor)
        sampled_ids, _ = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
    else:
        
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
    
    
    # Convert word_ids to words
    sampled_caption = ['<start>']
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print(sentence)
    
    with open(args.caption_path, 'r') as file:
        captions = json.load(file)
        
    caps = captions['annotations']    
    
    for image in captions['images']:
 
        if image['file_name']== args.image:
            image_id = image['id']
    list_cap = []
    for cap in caps:
        if cap['image_id'] == image_id:
            list_cap.append(cap['caption'])
    gts= {}
    res = {}
    gts[args.image] = list_cap
    res[args.image] = [sentence.strip('<start> ').strip(' <end>')]
        
    bleu_res = bleu(gts, res)
    cider_res = cider(gts, res)
    rouge_res = rouge(gts, res)

    image = Image.open(os.path.join(args.image_path, args.image))
    plt.imshow(np.asarray(image))
    return bleu_res, cider_res, rouge_res

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='VizWiz_val_00000001.jpg', help='input image for generating caption')
    parser.add_argument('--image_path', type=str, default = 'data/val/', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-att-8.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-att-8.ckpt', help='path for trained decoder')
    parser.add_argument('--model_path', type=str, default='models/decoder-att-test_1.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--caption_path', type=str, default='data/val.json', help='path for train annotation json file')
    parser.add_argument('--model_type', type=str , default='attention', help='no_attention or attention')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--transformer_layers', type=int , default= 6, help='number of transformer layers')
    parser.add_argument('--encoder_dim', type=int, default=2048, help = 'dimension of image embeddings image embeddings in attention model')
    parser.add_argument('--alpha_c', type=float, default= 1.0, help = 'coefficient to calculate attention loss')
    args = parser.parse_args()
    main(args)
