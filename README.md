# CryoFilter

## Problem Specification

What do the top ten most commonly prescribed medications, including treatments for hyperthyroidism, asthma, heart disease, and ADHD all have in common? They all target membrane proteins. In fact, despite membrane proteins making up only a third of proteins in the human body, over half of all medications target them including treatments currently in development for COVID-19. However, the structural and functional mechanisms of many of these proteins remains a mystery making the effective development of new medicines highly wasteful and difficult.

In 2017 the Nobel Prize in chemistry was awarded to Jacques Dubochet, Joachim Frank and Richard Henderson for their contributions to the development of cryo-electron microscopy, a technique that has led to a revolution in finding structures of membrane proteins by allowing researchers to capture 2D images of individual molecules and use these to create high resolution 3D models. These high resolution 3D models can then be used as the basis for a computational drug discovery platform. But despite these advances, building a high quality model of a membrane protein is not a trivial task.

All datasets have anomalies, in cryo-electron microscopy data these come in the form of broken particles. One key challenge in the analysis workflow is sorting high quality 2D images from images of broken particles or other false positives. Current platforms use a maximum likelihood approach to 2D classification that groups similar looking images into classes that can then be manually determined to be good, and used in further analysis, or bad and removed from the dataset. While useful on a very large scale, this method takes a significant amount of time and worse, it never truly removes all the bad images and just a few bad images in a dataset of thousands of particles can significantly hamper high quality 3D reconstruction.
This is where ‘CryoFilter’ will come in. We envisage a machine learning based software tool that can automatically detect and remove bad particles from the datasets, leaving only the good particles to rapidly create the 3D high resolution structure of the membrane protein. CryoFilter will integrate seamlessly into existing workflows while removing the crucial step of manual sorting. Once in use, CryoFilter will save biomedical researchers in hundreds of groups around the world weeks to months of time which they now spend manually selecting images, significantly speeding up the discovery time.

## TODO

- Exploratory data analysis of the image. See if we can construct features that tell us anything useful. Maybe these can be used as inputs to the classifiers.
- Develop binary classifiers to distinguish 'good' and 'bad' particles.
- If we develop multiple classifiers, ensemble them.

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

https://sphire.mpg.de/wiki/doku.php?id=pipeline:window:cryolo
