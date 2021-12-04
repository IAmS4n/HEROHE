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
@misc{condesousa2021herohe,
      title={HEROHE Challenge: assessing HER2 status in breast cancer without immunohistochemistry or in situ hybridization}, 
      author={Eduardo Conde-Sousa and João Vale and Ming Feng and Kele Xu and Yin Wang and Vincenzo Della Mea and David La Barbera and Ehsan Montahaei and Mahdieh Soleymani Baghshah and Andreas Turzynski and Jacob Gildenblat and Eldad Klaiman and Yiyu Hong and Guilherme Aresta and Teresa Araújo and Paulo Aguiar and Catarina Eloy and António Polónia},
      year={2021},
      eprint={2111.04738},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
