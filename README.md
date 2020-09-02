# Required packages:
Code is tested on Linux and python 3.5.2

For loading mrxs files, openslide library is used which needs to install some packages as bellow:

	sudo apt-get install openslide-tools
	
	sudo apt-get install python-openslide
	
Also, required python packages is listed in "requirement.txt".


# Usage:
There are two steps. First step is mask finding for slides, and the second step is prediction.

## First step:
Run following command in code directory:

	python3 test.py --test_dir TESTDIR --make_mask

## Second step:
Run following command in code directory:

	python3 test.py --test_dir TESTDIR --make_csv

Afterwards, a CSV file that contains results will be generated with the name "piaz.csv".
