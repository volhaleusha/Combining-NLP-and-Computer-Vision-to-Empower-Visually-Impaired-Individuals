import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import Encoder, Decoder, EncoderAtt, DecoderAtt, Transformer
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as T
from tqdm import tqdm
from evaluate import bleu, cider, meteor, rouge, spice
from tensorboardX import SummaryWriter
import json
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from utils import create_masks, load_image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    #create a writer
    writer = SummaryWriter('loss_plot_'+args.mode, comment='test')
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = T.Compose([ 
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    val_length = len(os.listdir(args.image_dir_val))
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    data_loader_val = get_loader(args.image_dir_val, args.caption_path_val, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the model
    # if no-attention model is chosen:
    if args.model_type == 'no_attention':
        encoder = Encoder(args.embed_size).to(device)
        decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
        criterion = nn.CrossEntropyLoss()
        
    # if attention model is chosen:
    elif args.model_type == 'attention':
        encoder = EncoderAtt(encoded_image_size=9).to(device)
        decoder = DecoderAtt(vocab, args.encoder_dim,  args.hidden_size, args.attention_dim,  
                 args.embed_size, args.dropout_ratio, args.alpha_c).to(device)
        
    # if transformer model is chosen:
    elif args.model_type == 'transformer': 
        model = Transformer(len(vocab), args.embed_size, args.transformer_layers , 8, args.dropout_ratio).to(device)
        
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                                 lr=args.learning_rate_enc)
        decoder_optimizer =torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                                 lr=args.learning_rate_dec)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])

    else:
        print('Select model_type attention or no_attention')
    
    # if model is not transformer: additional step in encoder is needed: freeze lower layers of resnet if args.fine_tune == True
    if args.model_type != 'transformer':
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=args.learning_rate_dec)
        encoder.fine_tune(args.fine_tune)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.learning_rate_enc)
        
    # initialize lists to store results: 
    loss_train = []
    loss_val = []
    loss_val_epoch = []
    loss_train_epoch = []
    
    bleu_res_list = []
    cider_res_list = []    
    rouge_res_list = []              
        
    results = {}
    
    # calculate total steps fot train and validation 
    total_step = len(data_loader)
    total_step_val = len(data_loader_val)
    
    #For each epoch
    for epoch in  tqdm(range(args.num_epochs)):

        loss_val_iter = []
        loss_train_iter = []
        
        
        # set model to train mode
        if args.model_type != 'transformer':
            encoder.train()
            decoder.train()
        else: 
            model.train()
        
        # for each entry in data_loader
        for i, (images, captions, lengths) in tqdm(enumerate(data_loader)):
            # load images and captions to device
            images = images.to(device)
            captions = captions.to(device)
            # Forward, backward and optimize
            
            # forward and backward path is different dependent of model type:
            if args.model_type == 'no_attention':
                # get features from encoder
                features = encoder(images)
                # pad targergets to a length
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                # get output from decoder
                outputs = decoder(features, captions, lengths)
                # calculate loss
                loss = criterion(outputs, targets)
              
                # optimizer and backward step
                decoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()            
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()
            
            elif args.model_type == 'attention':
                
                # get features from encoder
                features = encoder(images)
                
                # get targets - starting from 2 word in captions 
                #(the model not sequantial, so targets are predicted in parallel- no need to predict first word in captions)
                
                targets = captions[:, 1:] 
                # decode length = length-1 for each caption
                decode_lengths = [length-1 for length in lengths]
                #flatten targets
                targets = targets.reshape(targets.shape[0]*targets.shape[1])

                sampled_caption =[]
                
                # get scores and alphas from decoder
                scores,  alphas = decoder(features, captions, decode_lengths)

                scores = scores.view(-1, scores.shape[-1])
                
                #predicted = prediction with maximum score
                _, predicted = torch.max(scores, dim=1)

                # calculate loss 
                loss = decoder.loss(scores, targets, alphas)
                
                # optimizer and backward step
                decoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()            
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()
                
            elif args.model_type == 'transformer':
                
                # input is captions without last word
                trg_input = captions[:, :-1]
                # create mask
                trg_mask = create_masks(trg_input)
                
                # get scores from model
                scores = model(images, trg_input, trg_mask)
                scores = scores.view(-1, scores.shape[-1])
                
                # get targets - starting from 2 word in captions 
                targets = captions[:, 1:]
                
                #predicted = prediction with maximum score
                _, predicted = torch.max(scores, dim=1)
                
                # calculate loss 
                loss = criterion(scores, targets.reshape(targets.shape[0]*targets.shape[1]))

                #forward and backward path
                decoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()  
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()
            
            else:
                print('Select model_type attention or no_attention')
                
            # append results to loss lists and writer
            loss_train_iter.append(loss.item())
            loss_train.append(loss.item())
            writer.add_scalar('Loss/train/iterations', loss.item(), i+1)
            
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
        
        #append mean of last 10 batches as approximate epoch loss      
        loss_train_epoch.append(np.mean(loss_train_iter[-10:]))   

        
        writer.add_scalar('Loss/train/epoch', np.mean(loss_train_iter[-10:]), epoch+1)
        
        
        #save model
        if args.model_type != 'transformer':
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder_'+args.mode+'_{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'decoder_'+args.mode+'_{}.ckpt'.format(epoch+1)))
        
        else:
            torch.save(model.state_dict(), os.path.join(
                args.model_path, 'model_'+args.mode+'_{}.ckpt'.format(epoch+1)))
        np.save(os.path.join(args.predict_json, 'loss_train_temp_'+args.mode+'.npy'), loss_train)
        
        #validate model:
        # set model to eval mode:
        if args.model_type != 'transformer':
            encoder.eval()
            decoder.eval()
        else: 
            model.eval()
        total_step = len(data_loader_val)
        
        # set no_grad mode:
        with torch.no_grad():
             # for each entry in data_loader
            for i, (images, captions, lengths) in tqdm(enumerate(data_loader_val)): 
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                images = images.to(device)
                captions = captions.to(device)

                 # forward and backward path is different dependent of model type:
                if args.model_type == 'no_attention':
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)
                    loss = criterion(outputs, targets)

                elif args.model_type == 'attention':
                    
                    
                    features = encoder(images)
                    sampled_caption =[]
                    targets = captions[:, 1:]
                    decode_lengths = [length-1 for length in lengths]
                    targets = targets.reshape(targets.shape[0]*targets.shape[1])
                    
                    scores,  alphas = decoder(features, captions, decode_lengths)
                    
                    _, predicted = torch.max(scores, dim=1)
                    

                    scores = scores.view(-1, scores.shape[-1])

                    sampled_caption = []
                    
                    
                    loss = decoder.loss(scores, targets, alphas)
                    
                elif  args.model_type == 'transformer':
                    
                    trg_input = captions[:, :-1]
                    trg_mask = create_masks(trg_input)
                    scores = model(images, trg_input, trg_mask)
                    scores = scores.view(-1, scores.shape[-1])
                    targets = captions[:, 1:]

                    _, predicted = torch.max(scores, dim=1)
                    
                    loss = criterion(scores, targets.reshape(targets.shape[0]*targets.shape[1]))
                
                #display results 
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}, Validation Perplexity: {:5.4f}'
                          .format(epoch, args.num_epochs, i, total_step_val, loss.item(), np.exp(loss.item())))
                
                # append results to loss lists and writer
                loss_val.append(loss.item())
                loss_val_iter.append(loss.item())

                writer.add_scalar('Loss/validation/iterations', loss.item(), i+1)

        np.save(os.path.join(args.predict_json, 'loss_val_'+args.mode+'.npy'), loss_val)
                
        print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}, Validation Perplexity: {:5.4f}'
              .format(epoch, args.num_epochs, i, total_step_val, loss.item(), np.exp(loss.item())))
        
        # results: epoch validation loss

        loss_val_epoch.append(np.mean(loss_val_iter))
        writer.add_scalar('Loss/validation/epoch', np.mean(loss_val_epoch), epoch+1)
        
        #predict captions:
        filenames = os.listdir(args.image_dir_val)  

        predicted ={}

        for file in tqdm(filenames):
            if file  == '.DS_Store':
                continue
            # Prepare an image
            image = load_image(os.path.join(args.image_dir_val, file), transform)
            image_tensor = image.to(device)
            
            # Generate caption starting with <start> word
            
            # procedure is different for each model type
            if args.model_type == 'attention':
                
                features = encoder(image_tensor)
                sampled_ids, _ = decoder.sample(features)
                sampled_ids = sampled_ids[0].cpu().numpy()
                #start sampled_caption with <start>
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
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                # break at <end> of the sentence
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)

            predicted[file] = sentence
        
        # save predictions to json file: 
        json.dump(predicted, open(os.path.join(args.predict_json, 'predicted_'+args.mode+'_'+str(epoch)+'.json'), 'w'))    

        
        #validate model 
        with open(args.caption_path_val, 'r') as file:
            captions = json.load(file)
        
        res ={}
        for r in predicted:
            res[r] = [predicted[r].strip('<start> ').strip(' <end>')]
        
        images = captions['images']
        caps = captions['annotations']
        gts = {}
        for image in images:
            image_id = image['id']
            file_name = image['file_name']
            list_cap = []
            for cap in caps:
                if cap['image_id'] == image_id:
                    list_cap.append(cap['caption'])
            gts[file_name] = list_cap   
            
        #calculate BLUE, CIDER and ROUGE metrics from real and resulting captions    
        bleu_res = bleu(gts, res)
        cider_res = cider(gts, res)
        rouge_res = rouge(gts, res)
        
        # append resuls to result lists
        bleu_res_list.append(bleu_res)
        cider_res_list.append(cider_res)      
        rouge_res_list.append(rouge_res)             
        
        # write results to writer
        writer.add_scalar('BLEU1/validation/epoch', bleu_res[0], epoch+1)
        writer.add_scalar('BLEU2/validation/epoch', bleu_res[1], epoch+1)
        writer.add_scalar('BLEU3/validation/epoch', bleu_res[2], epoch+1)
        writer.add_scalar('BLEU4/validation/epoch', bleu_res[3], epoch+1)
        writer.add_scalar('CIDEr/validation/epoch', cider_res , epoch+1)
        writer.add_scalar('ROUGE/validation/epoch', rouge_res , epoch+1)
        
        
    results['bleu'] = bleu_res_list
    results['cider'] = cider_res_list
    results['rouge'] = rouge_res_list
    
    json.dump(results, open(os.path.join(args.predict_json, 'results_'+args.mode+'.json'), 'w'))
    np.save(os.path.join(args.predict_json, 'loss_train_'+args.mode+'.npy'), loss_train)
    np.save(os.path.join(args.predict_json, 'loss_val_'+args.mode+'.npy'), loss_val)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_test/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/Imagetest/train', help='directory for resized images')
    parser.add_argument('--image_dir_val', type=str, default='data/Imagetest/val', help='directory for resized images')
    parser.add_argument('--predict_json', default='output_test', help='folde with output json file')
    parser.add_argument('--caption_path', type=str, default='data/Imagetest/train.json', help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str, default='data/Imagetest/val.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--model_type', type=str , default='no_attention', help='no_attention or attention')
    
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=10, help = 'number of epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate_enc', type=float, default=0.001)
    parser.add_argument('--learning_rate_dec', type=float, default=0.001)
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--transformer_layers', type=int , default= 6, help='number of transformer layers')
    parser.add_argument('--encoder_dim', type=int, default= 2048, help = 'dimension of image embeddings image embeddings in attention model')
    parser.add_argument('--alpha_c', type=float, default= 1.0, help = 'coefficient to calculate attention loss')
    parser.add_argument('--fine_tune',  default= True, help = 'freeze lower layers of resnet')
    parser.add_argument('--mode',  default= 'test', help = 'define mode to save data')
    parser.add_argument('--encoded_image_size',  default= 9, help = 'define mode to save data')
    
    
    args = parser.parse_args()
    print(args)
    main(args)