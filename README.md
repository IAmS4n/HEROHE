# Required packages
Code is tested on Linux and python 3.5.2

For loading mrxs files, openslide library is used, which needs to install some packages as follows:

	sudo apt-get install openslide-tools
	
	sudo apt-get install python-openslide
	
Also, required python packages is listed in "requirement.txt".


# Usage
There are two steps: First, mask finding for slides. Second, prediction.

## First step:
Run following command in the code directory:

	python3 test.py --test_dir TESTDIR --make_mask

## Second step:
Run following command in the code directory:

	python3 test.py --test_dir TESTDIR --make_csv

As a result, a CSV file that contains results will be generated with the name of "piaz.csv".

# Citation
```latex
@Article{jimaging8080213,
AUTHOR = {Conde-Sousa, Eduardo and Vale, João and Feng, Ming and Xu, Kele and Wang, Yin and Della Mea, Vincenzo and La Barbera, David and Montahaei, Ehsan and Baghshah, Mahdieh and Turzynski, Andreas and Gildenblat, Jacob and Klaiman, Eldad and Hong, Yiyu and Aresta, Guilherme and Araújo, Teresa and Aguiar, Paulo and Eloy, Catarina and Polónia, Antonio},
TITLE = {HEROHE Challenge: Predicting HER2 Status in Breast Cancer from Hematoxylin&ndash;Eosin Whole-Slide Imaging},
JOURNAL = {Journal of Imaging},
VOLUME = {8},
YEAR = {2022},
NUMBER = {8},
ARTICLE-NUMBER = {213},
URL = {https://www.mdpi.com/2313-433X/8/8/213},
ISSN = {2313-433X},
ABSTRACT = {Breast cancer is the most common malignancy in women worldwide, and is responsible for more than half a million deaths each year. The appropriate therapy depends on the evaluation of the expression of various biomarkers, such as the human epidermal growth factor receptor 2 (HER2) transmembrane protein, through specialized techniques, such as immunohistochemistry or in situ hybridization. In this work, we present the HER2 on hematoxylin and eosin (HEROHE) challenge, a parallel event of the 16th European Congress on Digital Pathology, which aimed to predict the HER2 status in breast cancer based only on hematoxylin&ndash;eosin-stained tissue samples, thus avoiding specialized techniques. The challenge consisted of a large, annotated, whole-slide images dataset (509), specifically collected for the challenge. Models for predicting HER2 status were presented by 21 teams worldwide. The best-performing models are presented by detailing the network architectures and key parameters. Methods are compared and approaches, core methodologies, and software choices contrasted. Different evaluation metrics are discussed, as well as the performance of the presented models for each of these metrics. Potential differences in ranking that would result from different choices of evaluation metrics highlight the need for careful consideration at the time of their selection, as the results show that some metrics may misrepresent the true potential of a model to solve the problem for which it was developed. The HEROHE dataset remains publicly available to promote advances in the field of computational pathology.},
DOI = {10.3390/jimaging8080213}
}
```
