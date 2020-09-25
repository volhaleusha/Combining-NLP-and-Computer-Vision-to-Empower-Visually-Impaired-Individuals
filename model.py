import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)     
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)                
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    

class EncoderAtt(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=9):
        
        super(EncoderAtt, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-152

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))


    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim = 2048, 
                 decoder_dim = 512, 
                 attention_dim = 512):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        #print('att1', att1.size())
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        #print('att2', att2.size())
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        #print('att', att)
        alpha = self.softmax(att)  # (batch_size, num_pixels) 
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderAtt(nn.Module):

    def __init__(self, vocab, 
                 encoder_dim = 2048, 
                 decoder_dim = 512, 
                 attention_dim= 512,  
                 embed_size = 256, 
                 dropout_ratio = 0.5, 
                 alpha_c = 1, 
                 max_seq_length=20):
        
        super(DecoderAtt, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.vocab_size = len(vocab)
        self.dropout_ratio = dropout_ratio
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.alpha_c = alpha_c
        self.max_seg_length = max_seq_length

        self.attention = Attention( encoder_dim, decoder_dim, attention_dim )
        
        self.embedding = nn.Embedding(self.vocab_size, embed_size)  # embedding layer

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Gating scalars and sigmoid layer (cf. section 4.2.1 of the paper)
        self.f_beta = nn.Linear(decoder_dim,encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # LSTM
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_ratio)
        
        #self.fc = nn.Linear(decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary

        #loss Criterion
        self.criterion = nn.CrossEntropyLoss()
        
        self.linear_o = nn.Linear(embed_size, self.vocab_size)
        self.linear_h = nn.Linear(decoder_dim, embed_size)
        self.linear_z = nn.Linear(encoder_dim, embed_size)
        
        
    def init_hidden_states(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_output, encoded_captions, caption_lengths):
        
        
        """Perform a single decoding step."""
        
        batch_size = encoder_output.size(0)
        encoder_dim = encoder_output.size(-1)
        
        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_dim)
        
        #print(encoder_output.size())
        
        num_pixels = encoder_output.size(1)
        
        # Embedding
        embeddings = self.embedding(encoded_captions).type(torch.FloatTensor).to(device)
        #print(embeddings.size())
        
        #initial_states
        h, c = self.init_hidden_states(encoder_output)
        #print(h.size(), c.size())

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(caption_lengths), self.vocab_size) #.to(device)
        
        #print('prediction_length', predictions.size())
        alphas = torch.zeros(batch_size, max(caption_lengths), num_pixels) #.to(device)
        #print('alphas', alphas.size())
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(caption_lengths)):
            batch_size_t = sum([l > t for l in caption_lengths])
            att, alpha = self.attention(encoder_output[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            att = gate * att
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], att], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            #preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            
            h_embedded = self.linear_h(h)
            att_embedded = self.linear_z(att)
            preds = self.linear_o(self.dropout(embeddings[:batch_size_t, t, :] + h_embedded + att_embedded))
            
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        #print(predictions.size())    
        return predictions, alphas 
   
    def loss(self, outputs, targets, alphas):
        
        loss = self.criterion(outputs, targets.cpu())

        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        return loss

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""      

        #print('features', features.size())
        sampled_ids = []
 
        batch_size = features.size(0)
        
        encoder_dim = features.size(-1)
        
        features = features.view(batch_size, -1, encoder_dim)
        num_pixels = features.size(1)
        #print('features', features)

        prev_word = torch.LongTensor([[self.vocab.word2idx['<start>']]]).to(device)

        h, c = self.init_hidden_states(features)
        
        #print(h.size(), c.size())
        #print(h.mean())
        
        for t in range(self.max_seg_length):
            
            embeddings = self.embedding(prev_word).squeeze(1)
            #print(embeddings.size())
            att, _ = self.attention(features, h)
            #print('att', att)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            #print('gate', gate)
            att = gate * att
            #print('att', att.size())
            #print(h.size())
            h, c = self.decode_step(torch.cat([embeddings, att], dim=1), (h, c)) 
            #print(torch.cat([embeddings, att], dim=1))
            #print('h',h.mean())
            
            #print(h.size(), c.size())
            #preds = self.fc(self.dropout(h))  
            #print('preds', preds)
            
            h_embedded = self.linear_h(h)
            att_embedded = self.linear_z(att)
            preds = self.linear_o(self.dropout(embeddings + h_embedded + att_embedded))
            
            _, predicted = torch.max(preds, dim=1)
            #print(predicted)
            
            #print('indices', predicted)
            prev_word = predicted.unsqueeze(1)
            #print('prev', prev_word.size())
            
            sampled_ids.append(predicted)
            #print('sampled ids',  sampled_ids)
                       
        sampled_ids = torch.stack(sampled_ids, 1) 
        #print('ids', sampled_ids)

        
        return sampled_ids
    
