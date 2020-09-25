# Project Description
In this work several different approaches are explored to help visually impaired population by solving image captioning task for VizWiz dataset. The baseline model is based on CNN-LSTM architecture described with multiple modifications to boost performance. In addition, two different
architectures are explored: CNN-LSTM with attention based and CNN-transformer.



## Usage 

#### 1. Clone the repository
```bash
git clone https://github.com/volhaleusha/Combining-NLP-and-Computer-Vision-to-Help-Blind-People
```

#### 2. Download the dataset
```bash
python get_data.py
```
#### 3. Preprocessing

```bash
python build_vocab.py   
python resize.py
```

#### 4. Train the model

```bash
python train.py    
```

#### 5. Test the model 

```bash
python sample.py --image='png/example.png'
```
