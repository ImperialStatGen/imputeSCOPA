#ifndef __SCDATA_H__
#define __SCDATA_H__

#include <iostream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <armadillo>
#include <set>
#include <memory>

#include <Forest.h>
#include <ForestRegression.h>
#include <globals.h>
#include <utility.h>
#include <Data.h>
#include <Tree.h>
#include <TreeRegression.h>

#define to_uvec(x) arma::conv_to<arma::uvec>::from(arma::Col<size_t>(x))
#define ito_uvec(x) arma::conv_to<arma::uvec>::from(arma::Col<int>(x))

#define LOG(level, x) if(verbose > level) x;
#define SCOPA_MASK    

/* ArmaDouble is an implementation of Data meant for two things: 
   a.  to have arma::mat for its backend
   b.  to be re-usable so that it matches the SCOPA algorithm which calls Ranger repeatedly over the different columns
   In order to do that, it holds the imputed data matrix along with two vectors that act as masks.  
   
   Rather than shedding, we always use the full matrix but avoid the use of NA entries during growing and vice-versa during prediction.
*/
class ArmaDouble : public ranger::Data
{
 private:
  double    *rdata;
  
 public:
  arma::mat *pOG;
  arma::mat cdata;
  std::vector<arma::uvec> vecnas;   // NA Indices
  std::vector<arma::uvec> vecvals;  // Valid Indices (non-NA)
  std::vector<int>        arrange;  // This is just for rearrangement
  std::string             colname;
  arma::mat index_mat;
  arma::mat cwin;
  arma::umat cmap;
  int verbose;
  unsigned                maskcol;
  unsigned                pred_idx;
  
  ArmaDouble() = default;
  
  void prep(unsigned col, std::string cname) {
    pred_idx = 0;
    cdata.col(0) = pOG->col(col);
    maskcol = arrange[col];
    if (arrange[col] < 0) {
      maskcol = cdata.n_cols - 1;
      updateCMap(col, maskcol);
    }
    sort(0);
    cwin = cdata.rows(vecvals[col]);
    num_rows = vecvals[col].size(); 
    variable_names[0] = cname;
    colname = cname;
    no_split_variables.clear();
  }

  void prepPredict(unsigned col, std::string cname) {
    cdata.col(0) = arma::zeros<arma::vec>(cdata.n_rows);
    cwin = cdata.rows(vecnas[col]);
    num_rows = vecnas[col].size(); 
    variable_names[0] = cname;
    no_split_variables.clear();
    pred_idx = vecvals[col].size();
  }


  void init(size_t n_rows, size_t n_cols) {
    this->num_cols = 1;
    this->num_cols_no_snp = 1;
    //this->variable_names = variable_names;
    variable_names.push_back("empty");
    this->num_rows = n_rows;
    // Leave first column empty
    cdata.resize(n_rows, 1);
    //arma::vec zvec = arma::zeros<arma::vec>(n_rows);
    //cdata.insert_cols(0, zvec);
    arrange.resize(n_cols, -1);
    vecnas.resize(n_cols);
    vecvals.resize(n_cols);
    cmap.resize(n_rows, n_cols + 1);
  }

  void updateCMap(unsigned col, unsigned ccol) {
    // CMAP has the index if we're to look at it when imputing for that column
    std::vector<int> viota(vecvals[col].size()), niota(vecnas[col].size());
    std::iota(viota.begin(), viota.end(), 0);
    std::iota(niota.begin(), niota.end(), vecvals[col].size());
    arma::uvec vec(cdata.n_rows);
    vec(ito_uvec(viota)) = vecvals[col];
    vec(ito_uvec(niota)) = vecnas[col];
    cmap.col(ccol) = vec;
  }
  
  std::pair<arma::uvec, arma::uvec>
    insert (size_t col, std::string name, arma::vec v) {
    vecnas[col] = arma::find_nonfinite(v); //matv[i]);
    vecvals[col] = arma::find_finite(v); //matv[i]);
    if (vecnas[col].size() == 0) {
      arrange[col] = cdata.n_cols - 1;
      this->variable_names.push_back(name);
      cdata.insert_cols(cdata.n_cols, v);
      updateCMap(col, cdata.n_cols - 1);
      num_cols++;
      num_cols_no_snp++;
    }
    return std::make_pair (vecnas[col], vecvals[col]);
  }
  
  /* Internal for update of the cdata matrix */
  void chknAddCData (unsigned col)
  {
    if (arrange[col] == -1) {
      arrange[col] = cdata.n_cols - 1;
      variable_names.push_back(colname);
      cdata.insert_cols(cdata.n_cols, pOG->col(col));
      num_cols++;
      num_cols_no_snp++;
    }
  }

  void updateCData (unsigned col, arma::vec filler) 
  {
    chknAddCData(col);
    unsigned ccol = arrange[col] + 1;
    unsigned sz = filler.size();
    arma::uvec idxs = vecnas[col];
    LOG(1, std::cout<<"replaced "<<col<<"("<<ccol<<") : ");
    for (unsigned j = 0; j < sz; j++) {
      cdata(idxs[j], ccol) = filler(j);
      LOG(4, std::cout<<filler(j)<<",");
    }
    sort(ccol);
    LOG(3,std::cout<<std::endl);
  }
	
  ArmaDouble(const ArmaDouble&) = delete;
  ArmaDouble& operator=(const ArmaDouble&) = delete;
  virtual ~ArmaDouble() override = default;

  void sort(unsigned col);
  void sort() override;

  bool valid(size_t row) const {
    return !(arma::any(vecnas[maskcol] == row));
  }
  bool validIdx(size_t row) const {
    return row < cdata.n_rows;
  }

  size_t unmask (size_t row) const {
#ifdef SCOPA_MASK
    return cmap.at(row + pred_idx, maskcol);
#else
    return row;
#endif
  }
  void printIndices () {
    std::cout<<std::endl<<" IMAT: "<<std::endl;
    for (uint i = 0; i < cdata.n_rows; i++) {
      std::cout<<i<<": ";
      for (uint j = 0; j < cdata.n_cols; j++) {
	std::cout<<index_mat.at(i, j)<<",";
      }
      std::cout<<std::endl;
    }
  }
  size_t getIndex(size_t row, size_t col) const override {
    // Use permuted data for corrected impurity importance
    size_t col_permuted = col;
    if (col >= num_cols) {
      col = getUnpermutedVarID(col);
      row = getPermutedSampleID(row);
    }

    // CMAP : VALID -> REAL
    // UNIQUE: Unique values in REAL
    // INDEX-MAT: REAL -> Index in UNIQUE
    if (col < num_cols_no_snp) {
      row = unmask(row);
      return index_mat.at(row, col); //index_data[col * num_rows + row];
    } else {
      return getSnp(row, col, col_permuted);
    }
  }

  double get(size_t row, size_t col) const override {
    // Use permuted data for corrected impurity importance
    if (col >= num_cols) {
      col = getUnpermutedVarID(col);
      row = getPermutedSampleID(row);
    }
    if (col < num_cols_no_snp) {
#if 0 
      double val = cdata.at(unmask(row), col);
#else
      double val = cwin.at(row, col);
#endif
      return val;
    }

    //return cdata.at(row, col);
    size_t col_permuted = col;        
    return getSnp(row, col, col_permuted);
  }

  /*std::vector<double> mget(std::vector<size_t> rows, size_t col) {
    arma::vec colrows = arma::conv_to<arma::vec>::from(cwin.col(col)).elem(to_uvec(rows));
    return arma::conv_to<std::vector<double>>::from(arma::unique(colrows));
    }*/
  void reserveMemory() override {
    cdata.resize(num_rows, num_cols);
  }


    void set(size_t col, size_t row, double value, bool& error) override {
      //cdata.at(row, col) = value;
      std::cout<<"Set called "<<std::endl;
      //cwin.at(row, col) = value;
    }
    double getUniqueDataValue(size_t varID, size_t index) const override {
      // Use permuted data for corrected impurity importance
      if (varID >= num_cols) {
	varID = getUnpermutedVarID(varID);
      }

      if (varID < num_cols_no_snp) {
	return unique_data_values[varID][index];
      } else {
	// For GWAS data the index is the value
	return (index);
      }
    }
    
  size_t getNumUniqueDataValues(size_t varID) const override {
    // Use permuted data for corrected impurity importance
    if (varID >= num_cols) {
      varID = getUnpermutedVarID(varID);
    }

    if (varID < num_cols_no_snp) {
      return unique_data_values[varID].size();
    } else {
      // For GWAS data 0,1,2
      return (3);
    }
  }

};

#define TREE_SCOPA

class TreeSCOPA : public ranger::TreeRegression
{
 public:
  TreeSCOPA() = default;
  TreeSCOPA(const TreeSCOPA&) = delete;
  TreeSCOPA& operator=(const TreeSCOPA&) = delete;
  virtual ~TreeSCOPA() override = default;
  TreeSCOPA(std::unique_ptr<ranger::Data>& indata);

  // Create from loaded forest
  TreeSCOPA(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
	    std::vector<double>& split_values,
	    std::unique_ptr<ranger::Data>& indata);


  void allocateMemory() override;
  
  double estimate(size_t nodeID) {
    // Mean of responses of samples in node
#ifdef SID_VECTOR
    size_t num_samples_in_node = sampleIDs[nodeID].size();
#else
    size_t num_samples_in_node = end_pos[nodeID] - start_pos[nodeID];
#endif
    double sum_responses_in_node = sum_node;
    return (sum_responses_in_node / (double) num_samples_in_node);
  }
 private:
  ArmaDouble *rdata;
  double sum_node;
  bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) override;

  // Called by splitNodeInternal(). Sets split_varIDs and split_values.
  bool findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  void findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_samples_node,
      double& best_value, size_t& best_varID, double& best_decrease);
  void findBestSplitValueSmallQ(size_t nodeID, size_t varID,  size_t num_samples_node,
				double& best_value, size_t& best_varID, double& best_decrease, std::vector<double> possible_split_values,
				std::vector<double>& sums_right, std::vector<size_t>& n_right);
  void findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_samples_node,
				double& best_value, size_t& best_varID, double& best_decrease);

  void addImpurityImportance(size_t nodeID, size_t varID, double decrease);

  double computePredictionMSE();

  double computePredictionAccuracyInternal() override {
    size_t num_predictions = prediction_terminal_nodeIDs.size();
    double sum_of_squares = 0;
    for (size_t i = 0; i < num_predictions; ++i) {
      size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
      double predicted_value = split_values[terminal_nodeID];
      double real_value = data->get(oob_sampleIDs[i], dependent_varID);
      if (predicted_value != real_value) {
	sum_of_squares += (predicted_value - real_value) * (predicted_value - real_value);
	}
    }
    return (1.0 - sum_of_squares / (double) num_predictions);
  }

  void cleanUpInternal() override {
    counter.clear();
    counter.shrink_to_fit();
    sums.clear();
    sums.shrink_to_fit();
  }

  std::vector<size_t> counter;
  std::vector<double> sums;

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
    void initInternal(std::string status_variable_name) override {
      reinitInternal();
      
      // Sort data if memory saving mode
      if (!memory_saving_splitting) {
	data->sort();
      }
    }
    void reinitInternal() {
      // If mtry not set, use floored square root of number of independent variables
      if (mtry == 0) {
	unsigned long temp = sqrt((double) (num_variables - 1));
	mtry = std::max((unsigned long) 1, temp);
      }
      
      // Set minimal node size
      if (min_node_size == 0) {
	min_node_size = ranger::DEFAULT_MIN_NODE_SIZE_REGRESSION;
      }

      // SCOPA doesn't sort because it will sort once up front and then upon the addition of
      // a new column, just that column
      //data->sort();
    }
public:
    ForestSCOPA() = default;
    ForestSCOPA(const TreeSCOPA&) = delete;
    ForestSCOPA& operator=(const TreeSCOPA&) = delete;
    virtual ~ForestSCOPA() override = default;
    
    std::unique_ptr<ranger::Data> getData() {
      return std::move(this->data);
    }
    void reinit(std::string colname, unsigned seed, std::unique_ptr<ranger::Data> indata) {
      this->data = std::move(indata);
      prediction_mode = false;
      // Initialize random number generator and set seed
      if (seed == 0) {
	std::random_device random_device;
	random_number_generator.seed(random_device());
      } else {
	random_number_generator.seed(seed);
      }

      // Set number of samples and variables
      num_samples = data->getNumRows();
      num_variables = data->getNumCols();

      data->addNoSplitVariable(dependent_varID);

      this->mtry = 0;
      this->reinitInternal(); //status_variable_name);
      
      num_independent_variables = num_variables - data->getNoSplitVariables().size();
      trees.clear();
      std::vector<std::string> catvars;
      this->data->setIsOrderedVariable(catvars);
      /* FIXME: Do I need to adjust these?
	 this->alpha = alpha;
	 this->minprop = minprop;
      */

    }
    void setPredict() {
      prediction_mode = true;
        //num_threads = 4;
      this->num_variables = data->getNumCols();
      this->num_samples = data->getNumRows();
    };
    double getPredictionError() {
        //computePredictionErrorInternal();
        return overall_prediction_error;
    }
    void growInternal() {
      trees.reserve(num_trees);
      for (size_t i = 0; i < num_trees; ++i) {
#ifdef TREE_SCOPA
	trees.push_back(ranger::make_unique<TreeSCOPA>(data));
#else
	trees.push_back(ranger::make_unique<ranger::TreeRegression>());
#endif
      }
    }

#ifdef TREE_SCOPA
    void predictInternal(size_t sample_idx) override {
      if (predict_all || prediction_type == ranger::TERMINALNODES) {
	// Get all tree predictions
	for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
	  if (prediction_type == ranger::TERMINALNODES) {
	    predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
	  } else {
	    predictions[0][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
	  }
	}
      } else {
	// Mean over trees
	double prediction_sum = 0;
	for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
	  prediction_sum += getTreePrediction(tree_idx, sample_idx);
	}
	predictions[0][0][sample_idx] = prediction_sum / num_trees;
      }
    }

    double getTreePrediction(size_t tree_idx, size_t sample_idx) const {
      const auto& tree = dynamic_cast<const TreeSCOPA&>(*trees[tree_idx]);
      return tree.getPrediction(sample_idx);
    }
    
    size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
      const auto& tree = dynamic_cast<const TreeSCOPA&>(*trees[tree_idx]);
      return tree.getPredictionTerminalNodeID(sample_idx);
    }
#endif
    
#if 0
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
    void loadForest(size_t dependent_varID, size_t num_trees,
		    std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
		    std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
		    std::vector<bool>& is_ordered_variable) {
      
      this->dependent_varID = dependent_varID;
      this->num_trees = num_trees;
      data->setIsOrderedVariable(is_ordered_variable);
      
      // Create trees
      trees.reserve(num_trees);
      for (size_t i = 0; i < num_trees; ++i) {
	trees.push_back(
#ifdef TREE_SCOPA
			ranger::make_unique<TreeSCOPA>(forest_child_nodeIDs[i], forest_split_varIDs[i], forest_split_values[i], data)
						       
#else
			ranger::make_unique<ranger::TreeRegression>(forest_child_nodeIDs[i], forest_split_varIDs[i], forest_split_values[i])
#endif
			);
      }
      
      // Create thread ranges
      ranger::equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
    }
#endif

};

#endif
