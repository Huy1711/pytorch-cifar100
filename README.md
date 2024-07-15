# Pytorch-cifar100

Practice on cifar100 using pytorch. I modified ResNeXt50 model structure compared to original repository. 

## Requirements

```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```

Download the dataset, unzip to `data/` folder

## Checkpoints

Link to checkpoints [Google Drive link](https://drive.google.com/drive/folders/1LSL4R3GUBb8K7zr61po7cT_FcSzK60W8?usp=sharing)

## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. run tensorbard(optional)
Install tensorboard
```bash
$ mkdir runs
# Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 3. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train seresnext50 encoder-decoder (cangjie model)
$ python train.py -net seresnext50 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

### 4. test the model
Test the model using test.py
```bash
# use gpu to run seresnext50 encoder-decoder on test set
$ python test.py -net seresnext50 -weights <path_to_seresnext50-gru_weights_file> -gpu
```

## Reference papers

- ResNeXt [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- SE Block [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- CRNN [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

## Training Details

Hyperparameter settings: 
lr = 0.01, train for maximum 5 epochs with batchsize 64 without weight decay, Nesterov momentum of 0.9.

Number of layers in GRU module is 2, hidden dimension is 512 (feature maps channel dimension divided by 4). For more detail, please visit `/models/heads/` folder.

## Results

[Link to the report]()

## Tensorboard

[Link to the report]()