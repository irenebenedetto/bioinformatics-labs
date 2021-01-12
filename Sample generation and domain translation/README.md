# Sample generation and domain translation

## Sample generation with GAN

This code implents a GAN for each class in order to generate new samples starting from mRNA.txt file. I
Then, implements an upper sampling technique using SMOTE and compares the results obtained with:
- PCA projecting onto the firsts 2 principal components all the samples;
- TSE projecting onto the the axes all the samples.

## Domain translation with Autoencoders

The scripts implements a Variational Autoencoder on mRNA samples, one on meth samples and performs the domain translation using the encoder for mRNA and the decoder for math samples.
Again, the results are comapred with PCA an TSNE.


