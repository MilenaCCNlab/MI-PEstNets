
rm(list = ls())
library(abc)


  
# test_set_summary_statistics : summary statistics of the behavior of agents from the held out test set used to evaluate the neural network
# train_set_summary_statistics : summary statistics of the behavior of agents from the train set used to train the neural network
# train_set_parameters : parameters used to simulate data of the agents from the train set used to train the neural network



nagents <- 3000
map_estimates <- matrix(data=NA,nrow=nestimate,ncol=mod$npar)
median_estimates <- matrix(data=NA,nrow=nestimate,ncol=mod$npar)
mean_estimates <- matrix(data=NA,nrow=nestimate,ncol=mod$npar)


for (i in 1:nagents){
  if (i%%500==0){
    print(i)
  }
  rej<- abc(target=test_set_summary_statistics[i,], param=train_set_parameters, sumstat = train_set_summary_statistics, tol=0.05, method =
              "rejection")
  
  acceptedsamples <- as.data.frame(rej$unadj.values)
  map_estimates[i,] <-sapply(acceptedsamples, map_estimate, precision = 2^10*8,method='KernSmooth')
  median_estimates[i,] <-sapply(acceptedsamples, median)
  mean_estimates[i,] <-sapply(acceptedsamples, mean)
  
}





