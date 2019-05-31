#include <scopa.h>
#include <csample.h>

/* This setup serves a few things simultaenously:
   a.  make Arma mat out of the data
   b.  Store NA indices
   c.  Setup set of number of na's so we can sort them (visit_seq)
   d.  Make cdata 
*/
unsigned SCOPA::setup(std::vector<std::vector<double> > &vvdata)
{
    arma::uvec              naidx;
    arma::uvec              vidx;
    scopa::setseed(seed);
    unsigned cols =  vvdata.size();
    vars.resize(cols);
    visit_seq.clear();

    if (!cols || !vvdata[0].size()) {
        return 0;
    }

    adata = new ArmaDouble;
    
    adata->verbose = verbose;
    adata->init (vvdata[0].size(), cols);
    
    for (unsigned i = 0; i < cols; i++) {
        arma::vec v(vvdata[i]);
        data.insert_cols(i, v); //arma::vec(vvdata[i]));
        naidx = arma::find_nonfinite(v); //matv[i]);
        vidx = arma::find_finite(v); //matv[i]);
        std::pair<arma::uvec, arma::uvec> idxs = adata->insert (i, names[i], v);
        unsigned nas = idxs.first.size();
        if (nas > 0) {
            visit_seq.push_back(std::make_pair(nas, i));
        }
        else {
            completed.emplace(i);
        }
        arma::uvec validxs = idxs.second;
        vars[i] = var(v.elem(validxs));
    }
    adata->pOG = &data;
    std::sort(visit_seq.begin(), visit_seq.end());
    if (verbose > 1) {
        VISSEQ_t::iterator visIt;
        for (visIt = visit_seq.begin(); visIt != visit_seq.end(); visIt++) {
            std::cout<<names[visIt->second]<<"\t";
        }
        std::cout<<std::endl;
    }

    rangerData = std::unique_ptr<ArmaDouble>(adata);
    return visit_seq.size();
}    

void SCOPA::createForest(unsigned num_trees) 
{
    // Now, setup ranger's million arguments.  This is based on the fact
    // that SCOPA uses default ranger values:
    forest = ranger::make_unique<ForestSCOPA>();
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

    /* For reference:
       void init(std::string dependent_variable_name, MemoryMode memory_mode, std::unique_ptr<Data> input_data, uint mtry,
       std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
       uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
       const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
       bool predict_all, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
       PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth);
    */
    std::string colname = rangerData->getVariableNames().size() ? rangerData->getVariableNames()[0] : "empty";
    forest->init (colname, mmode, std::move(rangerData), mtry, output_prefix, num_trees, seed, num_threads,
                  impmode, min_node_size, status_variable_name, false, replace,
                  catvars, memory_saving, splitrule, false, sampleFraction, alpha, minprop, holdout,
                  ranger::RESPONSE, num_random_splits, order_snps, max_depth);
    rangerData = forest->getData();
}
    
/* This should happen at most once, but if we have no complete
   columns at all, we fall here, where we just sample the column with the least NAs */
void SCOPA::imputeUnivariate(unsigned col)
{
    arma::uvec idxs = adata->vecnas[col];
    arma::vec dcol = data.col(col);
    arma::vec cpy (dcol.elem(adata->vecvals[col]));
    unsigned nas = idxs.size();
    LOG(3, std::cout<<"ImputeUnivar "<<col<<"("<<nas<<") : ");
    /*for (unsigned j = 0; j < nas; j++) {
        LOG(4,std::cout<<idxs[j]<<",");
        cpy.shed_row(idxs[j] - j);
        }*/
    LOG(3,std::cout<<std::endl);

    std::string colname = names[col];
    adata->prep(col, colname);
    arma::vec okrep = scopa::sample<arma::vec>(cpy, idxs.size(), true);
    adata->updateCData (col, okrep);
}

double SCOPA::impute(unsigned col) {
    double pe = 0, penorm;
        
    // Remove all observations with NA's in that column
    std::string colname = names[col];
    adata->prep(col, colname);

    forest->reinit(colname, seed, std::move(rangerData));
    forest->run (false, true);
    pe = forest->getOverallPredictionError();
    //std::cout<<"PE "<<pe<<std::endl;
    
    penorm = pe / vars[col];
    std::string dcolname("dependent");
    adata->prepPredict(col, dcolname);
    
    forest->setPredict();
    forest->run (false, false);

    // Retake ownership of this
    rangerData = forest->getData();
    
    std::vector<std::vector<std::vector<double> > > preds = forest->getPredictions();

    LOG(3, std::cout<<"PE "<<pe<<" / "<<vars[col]<<"="<<penorm<<" : "<<preds.size()<<" and "<<preds[0][0].size()<<std::endl);
    if (verbose > 4) {
        for (uint i = 0; i < preds[0][0].size(); i++) {
            std::cout<<preds[0][0][i]<<", ";
        }
        std::cout<<std::endl;
    }
    adata->updateCData (col, preds[0][0]);
        
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
void SCOPA::run(int num_trees, unsigned maxiters) 
{
    bool crit = true;
    double mpe_last = arma::datum::inf;
    data_last = data;
    uint iter = 0;
        
    for (iter = 0; crit && iter < maxiters; iter++) {
        LOG(0, std::cout<<"Iter "<<iter<<std::endl);
        VISSEQ_t::iterator visIt;
        double mpe = 0;
        uint mpe_sz = 0;
        data_last = adata->cdata;
        for (visIt = visit_seq.begin(); visIt != visit_seq.end(); visIt++) {
            if (completed.empty()) {
                imputeUnivariate(visIt->second);
            }
            else {
                if (forest.get() == NULL) {
                    createForest(num_trees);
                }
                mpe += impute(visIt->second);
                mpe_sz++;
            }
            completed.emplace(visIt->second);
        }
        mpe /= mpe_sz; 
        crit = (mpe < mpe_last);
        LOG(0, std::cout<<"MPE "<<mpe<<" last MPE "<<mpe_last<<std::endl);
        mpe_last = mpe;
    }
    arma::mat *pOut = (iter == 1 || (iter == maxiters && crit)) ? &adata->cdata : &data_last;

    /* Rearrange for the same order as the data */
    for (uint i = 0; i < adata->arrange.size(); i++) {
        out_cdata.insert_cols(i, pOut->col(adata->arrange[i] + 1));
    }
}    
