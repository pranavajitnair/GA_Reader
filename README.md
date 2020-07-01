# GA_Reader
Implementation of the paper [Gated-Attention Readers for Text Comprehension](https://arxiv.org/pdf/1606.01549.pdf) in PyTorch.

### Dataset
This code only supports training and validation on the CNN and Daily Mail cloze style dataset.

Data can be dowloaded [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing) 

### Training 
Include all training data, validation data and testing data in the same folder, first training data followed by validation and testing data.

For training and testing the model run 
```
python train.py 
```

Optional Arguments
```
--epochs            number of epochs 
--iterations        iterations per epoch
--lr                learning rate for Adam optimizer
--char_size         size for character embeddings
--embed_size        size for glove embeddings
--char_hidden_size  hidden size for character GRU
--hidden_size       hidden size for document and query GRU
--use_char          whether to use character embeddings or not
--use_features      whether to use qe-comm features or not
--batch_size        batch size for training
--gru_layers        number of GRU layers for document and query
--train_file        file having all the data (training+validation+testing)
--training_size     number of training examples 
--dev_size          number og validation examples
--test_size         number of testing examples
```
