output <- read.table("../data/3phigh_missing.txt", header=T, sep = "\t", na.strings = "NA");
output <- output[,2:dim(output)[2]]