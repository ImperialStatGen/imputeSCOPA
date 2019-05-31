#include <scopa.h>
#include <scutils.h>

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
    std::string ifname, ofname("cout.txt");
    int maxiters = DEFAULT_MAXITERS;
    int num_trees = DEFAULT_NUMTREES;
    unsigned verbose = DEFAULT_VERBOSITY;
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

    scFile file;
    std::vector<std::vector<double> > vvdata = file.load(ifname, separator);
    
    SCOPA engine (std::vector<std::string>(file.names.begin()+1, file.names.end()), seed, verbose);
    if (!engine.setup(vvdata)) {
        std::cout<<"There is nothing to do here"<<std::endl;
        return 0;
    }
    engine.run(num_trees, maxiters);
    file.save(ofname, engine.out_cdata, separator, noids);
    
    return 0;
}



