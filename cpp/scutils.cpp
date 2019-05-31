#include <scutils.h>

std::vector<std::vector<double> > scFile::load (std::string ifname, char separator)
{
    std::ifstream file (ifname);
    std::string line;
    std::string token;
    getline(file, line);
    
    std::stringstream header_line_stream(line);
    while (getline(header_line_stream, token, separator)) {
        if (token[token.size()-1]=='\r') {
            token.erase(token.size()-1);
        }
        names.push_back(token);
    }
    
    std::vector<int> ids;
    std::vector<std::vector<double> > vvdata(names.size() - 1);
    
    while (getline(file, line)) {
        if (file.eof())
            break;
        
        std::stringstream line_stream(line);
        int v = 0;
        while (getline(line_stream, token, separator)) {
            if (token[token.size()-1]=='\r') {
                token.erase(token.size()-1);
            }
        
            if (!v) {
                ids.push_back(atoi(token.c_str()));
            }
            else {
                if (token == "NA") {
                    vvdata[v-1].push_back(nan("NA"));
                }
                else {
                    double val = strtod(token.c_str(), NULL);
                    vvdata[v-1].push_back(val);
                }
            }
            v++;
        }
    }
    return vvdata;
}

void scFile::save(std::string ofname, arma::mat &data, char sep, bool noids)
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


