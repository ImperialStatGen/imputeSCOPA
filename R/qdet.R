detach("package:ranger", unload=TRUE);
library.dynam.unload("ranger", system.file(package="ranger"));
remove.packages("ranger");
