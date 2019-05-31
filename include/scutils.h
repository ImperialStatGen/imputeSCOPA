#ifndef __SCUTILS_H__
#define __SCUTILS_H__

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <armadillo>

#define DEFAULT_SEPARATOR     '\t'

class scFile 
{
public:
    std::vector<std::string> names;
    
    std::vector<std::vector<double> > load (std::string ifname, char separator=DEFAULT_SEPARATOR);
    void save(std::string ofname, arma::mat &data, char sep=DEFAULT_SEPARATOR, bool noids=false);
};

#endif
