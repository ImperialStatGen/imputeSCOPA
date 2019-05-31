//#include <RcppArmadilloExtensions/sample.h>
#include <Rcpp.h>
#include "../../include/csample.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;

/*NumericVector csample_num( NumericVector x,
                           int size,
                           bool replace,
                           NumericVector prob = NumericVector::create()
                           ) {*/

// [[Rcpp::export]]
NumericVector csample_num( NumericVector x,
                           int size,
                           bool replace) 
{
    NumericVector ret = scopa::sample(x, size, replace);//, prob);
    return ret;
}

// [[Rcpp::export]]
void csetseed (int seed)
{
    scopa::setseed(seed);
}
