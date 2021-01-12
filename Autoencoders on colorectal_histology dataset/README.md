# Autoencoders on colon-rectal dataset for image generation

## Different autoencoder and different losses


This script builds an simple autoencoder for generating new images starting from the ones in the dataset: the model  trained with a MSE Loss tends to bleur images, and for this reason a Percetual Loss is added using a pre-trained VGG model.

Furthermore, an attempt of variational autoencoder is trained.

<img src="https://github.com/irenebenedetto/bioinformatics-labs/blob/main/imgs/autoencoders.png">

