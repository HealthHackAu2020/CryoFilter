# CryoFilter

A deep learning solution for high resolution protein structures.

## Problem Specification

What do the top ten most commonly prescribed medications, including treatments for hyperthyroidism, asthma, heart disease, and ADHD all have in common? They all target membrane proteins. In fact, despite membrane proteins making up only a third of proteins in the human body, over half of all medications target them including treatments currently in development for COVID-19. However, the structural and functional mechanisms of many of these proteins remains a mystery making the effective development of new medicines highly wasteful and difficult.

In 2017 the Nobel Prize in chemistry was awarded to Jacques Dubochet, Joachim Frank and Richard Henderson for their contributions to the development of cryo-electron microscopy, a technique that has led to a revolution in finding structures of membrane proteins by allowing researchers to capture 2D images of individual molecules and use these to create high resolution 3D models. These high resolution 3D models can then be used as the basis for a computational drug discovery platform. But despite these advances, building a high quality model of a membrane protein is not a trivial task.

All datasets have anomalies, in cryo-electron microscopy data these come in the form of broken particles. One key challenge in the analysis workflow is sorting high quality 2D images from images of broken particles or other false positives. Current platforms use a maximum likelihood approach to 2D classification that groups similar looking images into classes that can then be manually determined to be good, and used in further analysis, or bad and removed from the dataset. While useful on a very large scale, this method takes a significant amount of time and worse, it never truly removes all the bad images and just a few bad images in a dataset of thousands of particles can significantly hamper high quality 3D reconstruction.
This is where ‘CryoFilter’ will come in. We envisage a machine learning based software tool that can automatically detect and remove bad particles from the datasets, leaving only the good particles to rapidly create the 3D high resolution structure of the membrane protein. CryoFilter will integrate seamlessly into existing workflows while removing the crucial step of manual sorting. Once in use, CryoFilter will save biomedical researchers in hundreds of groups around the world weeks to months of time which they now spend manually selecting images, significantly speeding up the discovery time.

## Discussion notes

Day 1, Fri

- Regarding crYOLO, it doesn't really make sense to use YOLO on this problem. It will do a bad job when the anchor boxes overlap.
- We could just do some naive classification.
- We could do some feature engineering. First step PCA, looking at class averages, etc. Later maybe deep-learning methods (VAE).
- We can work on models separately then ensemble the results at the end.

Day 2, Sat

- PCA is not enough, the variance explained by the best components is small and the projection doesn't help distinguish. This is surprising, looking at the components we find disk-like features which make sense.

  ![pca](https://github.com/HealthHackAu2020/CryoFilter/blob/master/figures/pca_proj.png?raw=true)

  ![pca](https://github.com/HealthHackAu2020/CryoFilter/blob/master/figures/pca_plot.png?raw=true)

- Mahasen thinks that the spatial spectrum of the images might be informative.
- Turns out that the PCA features can actually discriminate. We realised that the PCA components probably correspond to common orientations of the macro-molecule. Because of this, the average of the projections is not very useful to discriminate between the classes (some components will be orthogonal to the data and look like a bad particle). You can see a clear distinction by taking the average of the squares over the classes.

  ![pca](https://github.com/HealthHackAu2020/CryoFilter/blob/master/figures/pca_plot_squared.png?raw=true)

- Now we just need to figure out the decision boundary to do classification.


- Using a [roberts filter](http://man.hubwiz.com/docset/Scikit-image.docset/Contents/Resources/Documents/api/skimage.filters.html#skimage.filters.roberts) on the
positive and negative classes, a naive CNN achieves a validation accurcy of approximately 75%. If we can cannot beat this brain dead benchmark, then we are doomed.
![Base Classifcation](https://github.com/HealthHackAu2020/CryoFilter/blob/master/figures/roberts_filter_classifcation.png?raw=true)

- VAE appears to be learning the average image for both classes, with the majority of noise in both sets being on the periferal regions of the image. Performs worse than a PCA at this stage. Need to consider a better way to partition the classes.

## Approach

## TODO

- Exploratory data analysis of the image. See if we can construct features that tell us anything useful. Maybe these can be used as inputs to the classifiers.
- Develop classifiers to distinguish 'good' (two classes top and side) and 'bad' particles.
- Train a VAE on the 'good` images.
- If we develop multiple classifiers we can ensemble them.

Extensions

- Image deionising?
- Particle localisation (segmentation) eg. fasterRCNN on top of the classifiers.
- Can we use the class averages to inform the the classifiers. Might find more about this during the EDA.

## Data

Sample data we are using:
https://drive.google.com/drive/folders/1-9g48x00LE8LZasuwgcjnztHRyHhQ_QJ?usp=sharing

Taken from this dataset:
https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10059/

## Useful links

Great intro to cryo-EM:
https://www.nature.com/articles/nmeth.3700

Current deep-learning methods:
https://sphire.mpg.de/wiki/doku.php?id=pipeline:window:cryolo
https://www.biorxiv.org/content/10.1101/838920v1.full

Availability:
https://docs.google.com/spreadsheets/d/1IBo6jAY8HO5a3b39HDeOO_1z1Blqu9jN21SIKq2kv3c/edit?usp=sharing

## Team Members

- Gavin Rice
- Simon Thomas
- George Li
- Henry Orton
- Mahasen Sooriyabandra
- Prithvi Reddy
- Issac Tucker
