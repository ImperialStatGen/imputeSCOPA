#ifndef __SCOPA_H__
#define __SCOPE_H__

#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <iomanip>

#include <stdlib.h>
#include <armadillo>
#include <set>

#include <Forest.h>
#include <ForestRegression.h>
#include <globals.h>
#include <utility.h>
#include <Data.h>
#include <Tree.h>

#include <scData.h>

#define DEFAULT_MAXITERS      5
#define DEFAULT_NUMTREES      10
#define DEFAULT_VERBOSITY     0

class SCOPA 
{
    typedef std::vector<std::pair<unsigned, unsigned> > VISSEQ_t;

public:
    std::vector<std::string> names;
    arma::mat               data;

    arma::mat               out_cdata;
    arma::mat               data_last;   
    // This is the current data (with whatever rows removed or
    // imputed, but always complete).  It's arranged differently,
    // correspondence is: 
    // cdata[:,arrange[col]] ~ data[:,col]
private:
    VISSEQ_t                visit_seq;
    std::set<unsigned >     completed;
    unsigned                seed;
    std::vector<double>     vars;  // Variances for computing the PE
    int verbose = 0;

    ArmaDouble             *adata;
    std::unique_ptr<ranger::Data> rangerData;
    std::unique_ptr<ForestSCOPA> forest;

    void createForest(unsigned num_trees);
public:
 SCOPA(std::vector<std::string> varnames, unsigned sd=0, unsigned verb=1) : names(varnames), seed(sd), verbose(verb) {
      LOG(1, std::cout<<"names "<<names.size()<<" : "<<names[0]<<std::endl);
    };
    ~SCOPA(){};

    /* The main user functions should be these two */
    unsigned setup(std::vector<std::vector<double> > &vvdata);
    void run(int num_trees, unsigned maxiters);

    /* The imputation methods corresponding to the R implementation */
    double impute(unsigned col);
    void imputeUnivariate(unsigned col);
    
};

#endif
