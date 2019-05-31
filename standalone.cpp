#include <iostream>
#include <fstream>
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
#include "csample.h"

#include <Forest.h>
#include <ForestRegression.h>
#include <globals.h>
#include <utility.h>
#include <Data.h>
#include <Tree.h>

#define DEFAULT_MAXITERS      5
#define DEFAULT_NUMTREES      10
#define DEFAULT_VERBOSITY     0
#define DEFAULT_SEPARATOR     '\t'

#define LOG(level, x) if(verbose > level) x;
    
static     int verbose = 0;

class scFile 
{
public:
    std::vector<std::string> names;
    
    std::vector<std::vector<double> > load (std::string ifname, char separator=DEFAULT_SEPARATOR) 
    {
        std::ifstream file (ifname);
        std::string line;
        std::string token;
        getline(file, line);
        LOG(5, std::cout<<"Header line "<<line<<std::endl);
        
        std::stringstream header_line_stream(line);
        while (getline(header_line_stream, token, separator)) {
            names.push_back(token);
        }
        LOG(1, std::cout<<"names "<<names.size()<<" : "<<names[0]<<std::endl);
        
        std::vector<int> ids;
        std::vector<std::vector<double> > vvdata(names.size() - 1);
    
        while (getline(file, line)) {
            if (file.eof())
                break;
        
            std::stringstream line_stream(line);
            int v = 0;
            while (getline(line_stream, token, separator)) {
                if (!v) {
                    ids.push_back(atoi(token.c_str()));
                }
                else {
                    if (token == "NA")
                        vvdata[v-1].push_back(nan("NA"));
                    else
                        vvdata[v-1].push_back(strtod(token.c_str(), NULL));
                }
                v++;
            }
        }
        return vvdata;
    }
    
    void save(std::string ofname, arma::mat &data, char sep=DEFAULT_SEPARATOR, bool noids=false)
    {
        std::ofstream file (ofname);

        std::vector<std::string>::iterator it;
        for (it = names.begin() + (int) noids; it != names.end(); it++) {
            file<<*it;
            if (it < names.end()-1)
                file<<sep;
        }
        
        file<<std::endl;

        file.precision(15);
        for (unsigned r = 0; r < data.n_rows; r++) {
            if (!noids)
                file<<(r+1)<<sep;
            for (unsigned c = 0; c < data.n_cols; c++) {
                file<<data(r,c);
                if (c < data.n_cols - 1)
                    file<<sep;
            }
            file<<std::endl;
        }
        file.flush();
    }
    
};

class ArmaDouble : public ranger::Data
{
private:
    arma::mat cdata;
    double    *rdata;
    
public:
    ArmaDouble() = default;
    ArmaDouble(std::vector<std::string> variable_names, arma::mat &in_cdata) : 
        cdata(in_cdata) {
        this->variable_names = variable_names;
        this->num_rows = in_cdata.n_rows;
        this->num_cols = in_cdata.n_cols;
        this->num_cols_no_snp = in_cdata.n_cols;
#if 0
        rdata = new double[this->num_rows * this->num_cols];
        for (uint r = 0; r < this->num_rows; r++) {
            for (uint c = 0; c < this->num_cols; c++) {
                rdata[c * this->num_rows + r] = cdata(r,c);
                
            }
        }
#endif
        LOG(3,std::cout<<"Constructing "<<in_cdata.n_rows<<"x"<<in_cdata.n_cols<<std::endl);
        LOG(3,std::cout<<"Var names "<<variable_names[0]<<std::endl);
        
    }

    ArmaDouble(const ArmaDouble&) = delete;
    ArmaDouble& operator=(const ArmaDouble&) = delete;

    virtual ~ArmaDouble() override = default;

    double get(size_t row, size_t col) const override {
        // Use permuted data for corrected impurity importance
        if (col >= num_cols) {
            col = getUnpermutedVarID(col);
            row = getPermutedSampleID(row);
        }
        if (col < num_cols_no_snp)
            return cdata.at(row, col);
            //return rdata[col * num_rows + row];
        size_t col_permuted = col;        
        return getSnp(row, col, col_permuted);
    }

    void reserveMemory() override {
        cdata.resize(num_rows, num_cols);
    }
    
    void set(size_t col, size_t row, double value, bool& error) override {
        //rdata[col * num_rows + row] = value;
        cdata.at(row, col) = value;
    }
};

class ForestSCOPA: public ranger::ForestRegression
{
private:
    size_t bak_trees;
    size_t bak_varID;
    
    std::vector<std::vector<std::vector<size_t>> > bak_child_nodeIDs;
    std::vector<std::vector<size_t>> bak_split_varIDs;
    std::vector<std::vector<double>> bak_split_values;
    std::vector<bool> bak_ordered_variable;
    
public:
    void setPredict(bool predict) {
        prediction_mode = predict;
        //num_threads = 4;
    };
    void setData (std::unique_ptr<ranger::Data> in_data) {
        this->data = std::move(in_data);
        this->num_variables = data->getNumCols();
        this->num_samples = data->getNumRows();
    };
    double getPredictionError() {
        //computePredictionErrorInternal();
        return overall_prediction_error;
    }
    void saveLocal()
    {
        bak_trees = num_trees;
        bak_varID = dependent_varID;
        std::vector<std::unique_ptr<ranger::Tree>>::iterator it;
        for (it = trees.begin(); it != trees.end(); it++) {
            bak_child_nodeIDs.push_back(it->get()->getChildNodeIDs());
            bak_split_varIDs.push_back(it->get()->getSplitVarIDs());
            bak_split_values.push_back(it->get()->getSplitValues());
        }
        bak_ordered_variable = getIsOrderedVariable();
    };
    void loadLocal() 
    {
        loadForest (bak_varID, bak_trees, bak_child_nodeIDs, bak_split_varIDs, bak_split_values,
                    bak_ordered_variable);
        minprop = 0;
        alpha = 0;
    }
};

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
    arma::mat               cdata;
    std::vector<std::string> cnames;
private:
    std::vector<int>        arrange;  // This is just for rearrangement
                                     
    std::vector<arma::uvec> vecnas;
    VISSEQ_t                visit_seq;
    std::set<unsigned >     completed;
    arma::uvec              naidx;
    unsigned                seed;
    std::vector<double>     vars;
        
public:
    SCOPA(std::vector<std::string> varnames, unsigned sd=0) : names(varnames), seed(sd) {};
    ~SCOPA(){};

    /* This setup serves a few things simultaenously:
       a.  make Arma mat out of the data
       b.  Store NA indices
       c.  Setup set of number of na's so we can sort them (visit_seq)
       d.  Make cdata 
    */
    unsigned setup(std::vector<std::vector<double> > &vvdata)
    {
        unsigned cols =  vvdata.size();
        unsigned compidx = 0;
        arrange.resize(cols);
        vecnas.resize(cols);
        vars.resize(cols);
        
        for (unsigned i = 0; i < cols; i++) {
            arma::vec v(vvdata[i]);
            data.insert_cols(i, v); //arma::vec(vvdata[i]));
            naidx = arma::find_nonfinite(v); //matv[i]);
            unsigned nas = naidx.size();
            if (nas > 0) {
                vecnas[i] = naidx;
                visit_seq.push_back(std::make_pair(nas, i));
                arrange[i] = -1;
            }
            else {
                completed.emplace(i);
                arrange[i] = compidx;
                cnames.push_back(names[i]);
                cdata.insert_cols(compidx++, v);
            }
            for (unsigned nar = 0; nar < nas; nar++) {
                v.shed_row(naidx[nar] - nar);
            }
            vars[i] = var(v);
        }

        std::sort(visit_seq.begin(), visit_seq.end());
        if (verbose > 1) {
            VISSEQ_t::iterator visIt;
            for (visIt = visit_seq.begin(); visIt != visit_seq.end(); visIt++) {
                std::cout<<names[visIt->second]<<"\t";
            }
            std::cout<<std::endl;
        }
        return visit_seq.size();
    }    

    
    /* This should happen at most once, but if we have no complete
       columns at all, we fall here, where we just sample the column with the least NAs */
    void imputeUnivariate(unsigned col)
    {
        arma::uvec idxs = vecnas[col];
        arma::vec cpy (data.col(col));
        unsigned nas = idxs.size();
        LOG(3, std::cout<<"ImputeUnivar "<<col<<"("<<nas<<") : ");
        for (unsigned j = 0; j < nas; j++) {
            LOG(4,std::cout<<idxs[j]<<",");
            cpy.shed_row(idxs[j] - j);
        }
        LOG(3,std::cout<<std::endl);
        
        arma::vec okrep = scopa::sample<arma::vec>(cpy, idxs.size(), true);
        updateCData (col, okrep);
    }

    void chknAddCData (unsigned col) 
    {
        if (arrange[col] == -1) {
            arrange[col] = cdata.n_cols;
            cnames.push_back(names[col]);
            cdata.insert_cols(cdata.n_cols, data.col(col));
        }
#if 0
        std::cout<<" Cdata : "<<std::endl;
        for (uint r = 0; r < cdata.n_rows; r++) {
            std::cout<<r<<": ";
            for (uint c = 0; c < cdata.n_cols; c++) {
                std::cout<<"("<<cdata(r, c)<<"::"<<data(r,c)<<"),";
            }
            std::cout<<std::endl;
        }
#endif   
    }
    void updateCData (unsigned col, arma::vec filler) 
    {
        chknAddCData(col);
        unsigned ccol = arrange[col];
        unsigned sz = filler.size();
        arma::uvec idxs = vecnas[col];
        LOG(1, std::cout<<"replaced "<<col<<"("<<ccol<<") : ");
        for (unsigned j = 0; j < sz; j++) {
            cdata(idxs[j], ccol) = filler(j);
            LOG(4, std::cout<<filler(j)<<",");
        }
        LOG(3,std::cout<<std::endl);
    }
        
    double impute(unsigned col, unsigned num_trees) {
        double pe = 0, penorm;
        
        // Remove all observations with NA's in that column
        std::string colname;
        arma::mat cpy;
        std::vector<std::string> inames;
#if 0
        chknAddCData(col);
        cpy = cdata;
        inames = cnames;
        colname = cnames[arrange[col]];
#else
        cpy.insert_cols(0, data.col(col));
        colname = names[col];
        inames.push_back(colname);
        for (uint c = 0; c < cdata.n_cols; c++) {
            cpy.insert_cols(c + 1, cdata.col(c));
            inames.push_back(cnames[c]);
        }
#endif
        arma::uvec idxs = vecnas[col];
        unsigned nas = idxs.size();
        arma::mat pred(nas, cpy.n_cols);
        LOG(3, std::cout<<"Impute "<<col<<"("<<nas<<") "<<"("<<names[col]<<","<<colname<<") : ");
        for (unsigned j = 0; j < nas; j++) {
            LOG(4, std::cout<<idxs[j]<<",");
            uint row = idxs[j] - j;
            pred(j, 0) = 0;
            pred.submat(j, 1, j, cdata.n_cols) = cdata.row(idxs[j]);
            cpy.shed_row(row);
        }
        LOG(3, std::cout<<std::endl);
        //std::cout.precision(16);
        LOG(3, std::cout<<"At this "<<col<<":"<<arrange[col]<<std::endl);
        
        // Now, setup ranger's million arguments.  This is based on the fact
        // that SCOPA uses default ranger values:
        std::unique_ptr<ranger::Forest> forest = ranger::make_unique<ForestSCOPA>();
        ranger::MemoryMode mmode(ranger::MEM_DOUBLE);
        uint mtry = 0;
        ranger::ImportanceMode impmode(ranger::IMP_NONE);
        uint num_threads = 0;
        uint min_node_size = 0;
        std::string output_prefix ("tst");
        std::string status_variable_name ("none");
        bool replace = true;
        bool memory_saving = false;
        ranger::SplitRule splitrule = ranger::LOGRANK;
        std::vector<double> sampleFraction(1);
        sampleFraction[0] = replace ? 1 : 0.632;
        double alpha = 0.5;
        double minprop = 0.1;
        bool holdout = false;
        uint num_random_splits = 1;
        bool order_snps = false;
        uint max_depth = 0;
        std::vector<std::string> catvars;
        std::unique_ptr<ranger::Data> rangerData = ranger::make_unique<ArmaDouble>(inames, cpy);
        
        LOG(3, std::cout<<"Current cdata "<<cnames.size()<<" col "<<col<<" : "<<arrange[col]<<":"<<colname<<std::endl);
        
        if (verbose > 4) {
            std::vector<std::string>::iterator it;
            for (it = cnames.begin(); it != cnames.end(); it++) {
                std::cout<<*it<<std::endl;
            }
            std::cout<<"Arranged: ";
            for (uint c = 0; c < arrange.size(); c++) {
                std::cout<<"("<<c<<":"<<arrange[c]<<"),";
            }
            std::cout<<std::endl;
        }
        /* For reference:
           void init(std::string dependent_variable_name, MemoryMode memory_mode, std::unique_ptr<Data> input_data, uint mtry,
           std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
           uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
           const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
           bool predict_all, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
           PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth);
        */
        forest->init (colname, mmode, std::move(rangerData), mtry, output_prefix, num_trees, seed, num_threads,
                      impmode, min_node_size, status_variable_name, false, replace,
                      catvars, memory_saving, splitrule, false, sampleFraction, alpha, minprop, holdout,
                      ranger::RESPONSE, num_random_splits, order_snps, max_depth);
        forest->run (false, true);
        pe = forest->getOverallPredictionError();
        penorm = pe / vars[col];
        inames[0] = "dependent";
        rangerData = ranger::make_unique<ArmaDouble>(inames, pred);
        (dynamic_cast<ForestSCOPA * >(forest.get()))->saveLocal();
        forest->init ("dependent", mmode, std::move(rangerData), mtry, output_prefix, num_trees, seed, num_threads,
                      impmode, min_node_size, status_variable_name, true, replace,
                      catvars, memory_saving, splitrule, false, sampleFraction, alpha, minprop, holdout,
                      ranger::RESPONSE, num_random_splits, order_snps, max_depth);
        (dynamic_cast<ForestSCOPA * >(forest.get()))->loadLocal ();
        forest->run (false, false);
        
        std::vector<std::vector<std::vector<double> > > preds = forest->getPredictions();

        LOG(3, std::cout<<"PE "<<pe<<" / "<<vars[col]<<"="<<penorm<<" : "<<preds.size()<<" and "<<preds[0][0].size()<<std::endl);
        if (verbose > 4) {
            for (uint i = 0; i < preds[0][0].size(); i++) {
                std::cout<<preds[0][0][i]<<", ";
            }
            std::cout<<std::endl;
        }
        updateCData (col, preds[0][0]);
        
        return std::isnan(penorm) ? 0 : penorm;
    }

    /* Starting point:  
       visit_seq - sorted by the number of NA's
       data - Data with missing observations
       completed - set of indices of data columns w/no missing observations
       
       The algorithm:
       1. If we have no complete columns, use univariate
       2. For each column col:
          a. Set cdata = data
          b. Remove all observations with na's in this column from
             cdata into mdata
          c. Build a ranger Forest from cdata
          d. Obtain range predictions for mdata
          e. Update cdata with range predictions
          f. Insert prediction Error into predErrors
       3. Calculate mean prediction error (MPE)
       4. Repeat from step 2 until MPE increases or we reach max iterations
    */
    void run(int num_trees, unsigned maxiters) 
    {
        bool crit = false;
        double mpe_last = arma::datum::inf;
        data_last = data;
        uint iter = 0;
        for (iter = 0; !crit && iter < maxiters; iter++) {
            LOG(0, std::cout<<"Iter "<<iter<<std::endl);
            VISSEQ_t::iterator visIt;
            double mpe = 0;
            uint mpe_sz = 0;
            data_last = cdata;
            for (visIt = visit_seq.begin(); visIt != visit_seq.end(); visIt++) {
                if (completed.empty()) {
                    imputeUnivariate(visIt->second);
                }
                else {
                    mpe += impute(visIt->second, num_trees);
                    mpe_sz++;
                }
                completed.emplace(visIt->second);
            }
            mpe /= mpe_sz; 
            crit = (mpe > mpe_last);
            LOG(0, std::cout<<"MPE "<<mpe<<" last MPE "<<mpe_last<<std::endl);
            mpe_last = mpe;
        }
        arma::mat *pOut = (iter == 1 || (iter == maxiters && crit)) ? &cdata : &data_last;

        /* Rearrange for the same order as the data */
        for (uint i = 0; i < arrange.size(); i++) {
            out_cdata.insert_cols(i, pOut->col(arrange[i]));
        }
    }    
};

void usage(char *prog) 
{
    std::cout<<prog<<": Missing value imputation by chained tree ensembles"<<std::endl;
    std::cout<<"Usage: "<<prog<<" -i <input_file> [-o <output_file>] [-m <maxiters>] [-n <num_trees>] [-v <verbosity>] [-h]"<<std::endl;
    std::cout<<"-i input_file:  File with missing data, required"<<std::endl;
    std::cout<<"-o output_file: File for the output data, default: out.txt"<<std::endl;
    std::cout<<"-I Don't print ID column (for comparing the imputeSCOPA.R output) "<<std::endl;
    std::cout<<"-m maxiters: Maximum number of iterations, default: "<<DEFAULT_MAXITERS<<std::endl;
    std::cout<<"-n num_trees: Number of trees, default: "<<DEFAULT_NUMTREES<<std::endl;
    std::cout<<"-s separator: separator character, for tab don't pass this argument "<<std::endl;
    std::cout<<"-S seed: seed for RNG (default is time)"<<std::endl;
    std::cout<<"-v verbosity: verbosity level, default "<<DEFAULT_VERBOSITY<<std::endl;
    std::cout<<"-h: show this usage"<<std::endl;
}

int main (int argc, char *argv[])
{
    int opt;
    std::string ifname, ofname("out.txt");
    int maxiters = DEFAULT_MAXITERS;
    int num_trees = DEFAULT_NUMTREES;
    verbose = DEFAULT_VERBOSITY;
    char separator = DEFAULT_SEPARATOR;
    int seed = 0;
    bool noids = false;
            
    while ((opt = getopt(argc,argv,"i:o:m:n:v:s:S:hI")) != EOF) {
        switch (opt) {
        case 'h':
            usage(argv[0]);
            return 0;
        case 'i':
            ifname = optarg;
            break;
        case 'o':
            ofname = optarg;
            break;
        case 'n':
            num_trees = atoi(optarg);
            break;
        case 'm':
            maxiters = atoi(optarg);
            break;
        case 'v':
            verbose = atoi(optarg);
            break;
        case 's':
            separator = optarg[0];
            break;
        case 'S':
            seed = atoi(optarg);
            break;
        case 'I':
            noids = true;
            break;
        default:
            usage(argv[0]);
            return -1;
        }
    }
    if (ifname == "") {
        std::cout<<"Missing arguments"<<std::endl;
        usage(argv[0]);
        return -1;
    }

    scopa::setseed(seed);

    scFile file;
    std::vector<std::vector<double> > vvdata = file.load(ifname, separator);
    
    SCOPA engine (std::vector<std::string>(file.names.begin()+1, file.names.end()), seed);
    if (!engine.setup(vvdata)) {
        std::cout<<"There is nothing to do here"<<std::endl;
        return 0;
    }
    engine.run(num_trees, maxiters);
    file.save(ofname, engine.out_cdata, separator, noids);
    
    return 0;
}



