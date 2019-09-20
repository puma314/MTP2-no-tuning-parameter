library(flare)
library(RcppCNPy)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/tiger_in_%s.npy", uid)
#inp = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/tiger_in.npy")
X <- npyLoad(inp, dotranspose=TRUE)
print("Loaded X")
d <- NCOL(X)
n <- NROW(X)
opt_lambda <- 3.1415 * sqrt(log(d)/n)
res <- sugm(data=X,
            lambda=opt_lambda,
            method='tiger',
            standardize=FALSE,
            perturb=FALSE)

print("DONE with tiger")
cov<-res$icov[[1]]
#out = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/tiger_out.npy")

out = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/tiger_out_%s.npy", uid)
npySave(out, cov)
