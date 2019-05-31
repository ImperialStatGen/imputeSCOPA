imputeSCOPA: a C++ tool for imputing high-dimensional data.
Currently imputeSCOPA implements the random forest algorithm for imputation. 

Requirements:
-------------
1. armadillo   http://arma.sourceforge.net/download.html
2. cmake

Organization:
-------------
CMakeLists.txt, standalone.cpp, cpp, include, ranger: necessary files and directories to build imputeSCOPA. 

R: includes csImputeSCOPA, a modified version of imputeSCOPA that samples from the same random generator so that the results from the C++ tool and R tool can be compared.

data: includes an example dataset to run the software.

To build:
---------
$export RANGER_DIR=/path/to/your/installation/ranger 
$mkdir build
$cd build 
$cmake .. 
$make

You should get an imputeSCOPA executable. The options for usage can be seen by just running the command ./imputeSCOPA. 

To run:
-------
a. imputeSCOPA (C++):
./imputeSCOPA -i ../data/3phigh_missing.txt -S <seed> -v 1 -m 10 -o ../data/cout.txt

b. R version:
source('csImputeSCOPA.R')
imp <- csImputeSCOPA(output, maxiter = 5, num.trees = 10, verbose = TRUE, seed=<seed>)
write.table(data.frame("ID"=rownames(imp),imp), "../data/tstout.txt",  quote=F, row.names=F, sep="\t");

If you are looking for duplication, you need to use the same seed in both a and b. 

