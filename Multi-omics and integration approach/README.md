# Multi-omics dataset integration apprach

# Multi-omics classification

Using the three datasets mRNA.txt, meth.txt, prot.txt (which contain respectively transcriptome, genome and proteome of a simulated dataset) and the file clusters.txt the scripts compares early and late integration approaches for sample classification.
- for both approaches implement 4 different classifiers. One of these must output not only the class label but also its probability;
- balance the classes and in the feature selection/dimensionality reduction process.

# Multi-omics clustering

Using the three datasets mRNA.txt, meth.txt, prot.txt (which contain respectively transcriptome, genome and proteome of a simulated dataset) and the file clusters.txt the scripts compares early and late integration approaches for sample clustering.
- for both approaches implement 4 different classifiers. One of these must output not only the class label but also its probability;
- balance the classes and in the feature selection/dimensionality reduction process.

In this case the algorithm for clustering is the MLP, proposend  in paper: <a href=https://link.springer.com/chapter/10.1007/978-3-662-12433-8_8>A Multilayer Perceptron for Clustering</a>.

The loss in this case is the union of the following losses:

<img src="https://github.com/irenebenedetto/bioinformatics-labs/blob/main/imgs/MLPclustering.png">

