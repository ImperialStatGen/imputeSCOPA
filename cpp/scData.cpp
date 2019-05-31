#include <scData.h>

#include <algorithm>
#include <iostream>
#include <iterator>

#include <ctime>

#include "assert.h"
#include "utility.h"
#include "TreeRegression.h"
#include "Data.h"

void ArmaDouble::sort(unsigned col) {
#ifdef SCOPA_MASK
    arma::mat *pData = &cdata;
#else
    arma::mat *pData = &cwin;
#endif
    
    if (col >= unique_data_values.size()) {
        index_mat.resize(pData->n_rows, num_cols_no_snp);
        unique_data_values.resize(num_cols_no_snp);
    }

    index_mat.col(col) = arma::zeros<arma::vec>(pData->n_rows);
       
    // Get all unique values
    std::vector<double> unique_values(pData->n_rows);
    for (size_t row = 0; row < pData->n_rows; ++row) {
        unique_values[row] = pData->at(row, col); 
    }
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(unique(unique_values.begin(), unique_values.end()), unique_values.end());

    // 
    // Get index of unique value
    for (size_t row = 0; row < pData->n_rows; ++row) {
        size_t idx = std::lower_bound(unique_values.begin(), unique_values.end(), pData->at(row, col)) - unique_values.begin(); //get(row,
                                                                                                                                //col))
                                                                                                                                //-
                                                                                                                                //unique_values.begin();
        index_mat.at(row, col) = idx; //index_data[col * num_rows +
                                      //row] = idx;
    }

    
    // Save unique values
    unique_data_values[col] = unique_values;
    if (unique_values.size() > max_num_unique_values) {
        max_num_unique_values = unique_values.size();
    }
}

/* Sort the entire thing and do the uniqueness lookup on the whole
 * matrix but what we return on get's is masked
 */
void ArmaDouble::sort() {
#ifndef SCOPA_MASK
    if (!cwin.n_cols)
        return;
#endif
    
    // For all columns, get unique values and save index for each observation
    unique_data_values.clear();
    
    // For all columns, get unique values and save index for each observation
    for (size_t col = 0; col < num_cols_no_snp; ++col) {
        sort(col);
    }
}

TreeSCOPA::TreeSCOPA (std::unique_ptr<ranger::Data>& indata) : TreeRegression()
{
    rdata = dynamic_cast<ArmaDouble *>(indata.get());
    for (size_t varID = 0; varID < data->getNumCols(); varID++) {

    }
}


TreeSCOPA::TreeSCOPA(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
                     std::vector<double>& split_values,
                     std::unique_ptr<ranger::Data>& indata) : 
    ranger::TreeRegression(child_nodeIDs, split_varIDs, split_values){
    rdata = dynamic_cast<ArmaDouble *>(indata.get());
}

void TreeSCOPA::allocateMemory() {        
  // Init counters if not in memory efficient mode
    if (1) {//!memory_saving_splitting) {
    size_t max_num_splits = data->getMaxNumUniqueValues();

    // Use number of random splits for extratrees
    if (splitrule == ranger::EXTRATREES && num_random_splits > max_num_splits) {
      max_num_splits = num_random_splits;
    }

    counter.resize(max_num_splits);
    sums.resize(max_num_splits);
  }
}

//std::cout<<"Res 1: "<<sum_responses_in_node<<" vs "<<sum_resp2<<std::endl;

bool TreeSCOPA::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
#ifdef SID_VECTOR
    uint sz = sampleIDs[nodeID].size();
#else
    uint sz = end_pos[nodeID] - start_pos[nodeID];
#endif
    bool stop = (sz <= min_node_size);  
    
  if (!stop) {      
      // Check if node is pure and set split_value to estimate and stop if pure
      bool pure = true;
      double pure_value = 0;

#ifdef SID_VECTOR
      for (size_t i = 0; i < sampleIDs[nodeID].size(); ++i) {
          double value = data->get(sampleIDs[nodeID][i], dependent_varID);
          if (i != 0 && value != pure_value) {
#else
      for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
          size_t sampleID = sampleIDs[pos];
          double value = data->get(sampleID, dependent_varID);
          if (pos != start_pos[nodeID] && value != pure_value) {
#endif
              pure = false;
              break;
          }
          pure_value = value;
      }
      
      if (pure) {
          split_values[nodeID] = pure_value;
          return true;
      }
  }
  sum_node = 0;
#ifdef SID_VECTOR
  for (auto& sampleID : sampleIDs[nodeID]) {
#else
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
#endif
      sum_node += data->get(sampleID, dependent_varID);
  }
  if (!stop) {
      // Find best split, stop if no decrease of impurity
      stop = findBestSplit(nodeID, possible_split_varIDs);
  }
        
  if (stop) {
      split_values[nodeID] = estimate(nodeID);
      return true;
  }

  return false;
}



bool TreeSCOPA::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
#ifdef SID_VECTOR
    uint num_samples_node = sampleIDs[nodeID].size();
#else
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
#endif
    double best_decrease = -1;
    size_t best_varID = 0;
    double best_value = 0;
    
    // For all possible split variables
    for (auto& varID : possible_split_varIDs) {
        
        // Find best split value, if ordered consider all values as split values, else all 2-partitions
        //if (data->isOrderedVariable(varID)) {
        // Use faster method for both cases
        double q = (double) num_samples_node / (double) data->getNumUniqueDataValues(varID);
        if (q < ranger::Q_THRESHOLD) {
            findBestSplitValueSmallQ(nodeID, varID, num_samples_node, best_value, best_varID, best_decrease);
        } else {
            findBestSplitValueLargeQ(nodeID, varID, num_samples_node, best_value, best_varID, best_decrease);
        }
    }
    
// Stop if no good split found
    if (best_decrease < 0) {
        return true;
    }
    
// Save best values
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;
    
// Compute decrease of impurity for this node and add to variable importance if needed
    if (importance_mode == ranger::IMP_GINI || importance_mode == ranger::IMP_GINI_CORRECTED) {
        addImpurityImportance(nodeID, best_varID, best_decrease);
    }
    return false;
}

void TreeSCOPA::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
    // Create possible split values
  std::vector<double> possible_split_values;
#ifdef SID_VECTOR
  data->getAllValues(possible_split_values, sampleIDs[nodeID], varID);
#else
  data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);
#endif
  
  // Try next variable if all equal for this
  if (possible_split_values.size() < 2) {
    return;
  }

  // -1 because no split possible at largest value
  const size_t num_splits = possible_split_values.size() - 1;
  std::fill_n(sums.begin(), num_splits, 0);
  std::fill_n(counter.begin(), num_splits, 0);
  findBestSplitValueSmallQ(nodeID, varID, num_samples_node, best_value, best_varID, best_decrease,
                           possible_split_values, sums, counter);
}

void TreeSCOPA::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_samples_node,
                                         double& best_value, size_t& best_varID, double& best_decrease, std::vector<double> possible_split_values,
                                         std::vector<double>& sums_right, std::vector<size_t>& n_right) {
  // -1 because no split possible at largest value
  const size_t num_splits = possible_split_values.size() - 1;

  // Sum in right child and possbile split
#ifdef SID_VECTOR
  for (auto& sampleID : sampleIDs[nodeID]) {
#else
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
#endif
      double value = data->get(sampleID, varID);
      double response = data->get(sampleID, dependent_varID);
      
      // Count samples until split_value reached
      for (size_t i = 0; i < num_splits; ++i) {
          if (value > possible_split_values[i]) {
              ++n_right[i];
              sums_right[i] += response;
          } else {
              break;
          }
      }
  }
  
  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < num_splits; ++i) {
      
      // Stop if one child empty
      size_t n_left = num_samples_node - n_right[i];
      if (n_left == 0 || n_right[i] == 0) {
          continue;
      }
      
      double sum_right = sums_right[i];
      double sum_left = sum_node - sum_right;
      double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right[i];
      
      // If better than before, use this
      if (decrease - best_decrease > 1e-5) {
          //if (decrease > best_decrease) {
          best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
          best_varID = varID;
          best_decrease = decrease;
          
          // Use smaller value if average is numerically the same as the larger value
          if (best_value == possible_split_values[i + 1]) {
              best_value = possible_split_values[i];
          }
      }
  }
}
  

void TreeSCOPA::findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_samples_node,
                                         double& best_value, size_t& best_varID, double& best_decrease) {

  // Set counters to 0
  size_t num_unique = data->getNumUniqueDataValues(varID);
  std::fill_n(counter.begin(), num_unique, 0);
  std::fill_n(sums.begin(), num_unique, 0);
  
#ifdef SID_VECTOR
  for (auto& sampleID : sampleIDs[nodeID]) {
#else
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
#endif
      size_t idx = data->getIndex(sampleID, varID);
      double value = data->get(sampleID, dependent_varID);
      sums[idx] += value;
      ++counter[idx];
  }
  
  size_t n_left = 0;
  double sum_left = 0;

  // Compute decrease of impurity for each split
  for (size_t i = 0; i < num_unique - 1; ++i) {
    // Stop if nothing here
    if (counter[i] == 0) {
      continue;
    }
    
    n_left += counter[i];
    sum_left += sums[i];

    // Stop if right child empty
    size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    double sum_right = sum_node - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // If better than before, use this
    if (decrease > best_decrease) {
      // Find next value in this node
      size_t j = i + 1;
      while (j < num_unique && counter[j] == 0) {
              ++j;
      }

      // Use mid-point split
      best_value = (data->getUniqueDataValue(varID, i) + data->getUniqueDataValue(varID, j)) / 2;
      best_varID = varID;
      best_decrease = decrease;

      // Use smaller value if average is numerically the same as the larger value
      if (best_value == data->getUniqueDataValue(varID, j)) {
        best_value = data->getUniqueDataValue(varID, i);
      }
    }
  }
}

void TreeSCOPA::addImpurityImportance(size_t nodeID, size_t varID, double decrease) {
#ifdef SID_VECTOR
    uint num_samples_node = sampleIDs[nodeID].size();
#else
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
#endif

    double best_decrease = decrease;
    if (splitrule != ranger::MAXSTAT) {
        best_decrease = decrease - sum_node * sum_node / (double) num_samples_node;
    }
    
    // No variable importance for no split variables
    size_t tempvarID = data->getUnpermutedVarID(varID);
    for (auto& skip : data->getNoSplitVariables()) {
        if (tempvarID >= skip) {
            --tempvarID;
        }
    }
    
    // Subtract if corrected importance and permuted variable, else add
    if (importance_mode == ranger::IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
        (*variable_importance)[tempvarID] -= best_decrease;
    } else {
        (*variable_importance)[tempvarID] += best_decrease;
    }
}


