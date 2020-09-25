# Project Description
In this work several different approaches are explored to help visually impaired population by solving image captioning task for VizWiz dataset. The baseline model is based on CNN-LSTM architecture described with multiple modifications to boost performance. In addition, two different
architectures are explored: CNN-LSTM with attention based and CNN-transformer.

## Usage 

#### 1. Clone the repository
```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
git clone https://github.com/volhaleusha/Combining-NLP-and-Computer-Vision-to-Help-Blind-People
```

#### 2. Download the dataset
1. 
```bash
# download VizWiz-Captioning dataset
python get_data.py

#download MSCOCO dataset
pip install -r requirements.txt
chmod +x download.sh
./download.sh
```
2. Download pretrained models folder from https://drive.google.com/drive/u/0/folders/16hUP6qbARpodivbIVDsK-516CB2vIxNL

#### 3. Preprocessing

```bash
python build_vocab.py   
python resize.py
```

#### 4. Train the model

```bash
python train.py --model_type 'no_attention'
```
Choose model type: 'no_attention', 'attention' or 'transformer'    

#### 5. Test the model 

```bash
python sample.py
```
Example (for more examples, check Visualization-from-paper.ipynb)
```bash
python sample.py --image 'data/Images/val/VizWiz_val_00000005.jpg' --image_path 'data/val' --encoder_path 'models/encoder1-4.ckpt' --decoder_path 'models/decoder1-4.ckpt' --vocab_path 'data/vocab.pkl' --num_layers 1 --model_type 'no_attention'
```
#### 6. Visualize attention for LSTM with Attention

```bash
python visualize_att.py 
```
Example (for more examples, check Visualization-from-paper.ipynb)
```bash
python visualize_att.py --image 'data/Images/val/VizWiz_val_00001623.jpg' --encoder_path 'models/encoder-att-8.ckpt' --decoder_path 'models/decoder-att-8.ckpt'
```
#### 7. Evaluate the model on BLUE, CIDEr and ROUGE metrics
```bash
python evaluate.py'
```

Example (for more examples, Analysis_from_paper.ipynb)
```bash
python evaluate.py --target_path 'data/val.json' --predict_path 'output/predicted_att_8.json'
```

