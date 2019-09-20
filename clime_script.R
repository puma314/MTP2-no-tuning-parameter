library(flare)
library(RcppCNPy)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/clime_in_%s.npy", uid)
X <- npyLoad(inp, dotranspose=TRUE)
print("Loaded X")
d <- NCOL(X)
n <- NROW(X)
clime_opt_lambda <-sqrt(log(d)/n)
res <- sugm(data=X,
            method='clime',
            standardize=FALSE,
            perturb=FALSE)

print("DONE with clime")
cov<-res$icov[[1]]

out = sprintf("/Users/umaroy/Documents/meng/MTP2-algorithm/clime_out_%s.npy", uid)
npySave(out, cov)
