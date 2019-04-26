library(sna)
library(ggplot2)
library(tidyverse)
library(rstudioapi) # load it
# the following line is for getting the path of your current open file
current_path <- getActiveDocumentContext()$path 
# The next line set the working directory to the relevant one:
setwd(dirname(current_path ))


######print supervised learning pictures

###random patterns
for(i in 1:5){
  
  figure<-rep(0,16)
  figure[sample(length(figure),4)]<-1
  png(paste0("../figures/1d_random_supervised_learning_",i,".png"),width=150, height=400)
    plot.sociomatrix(matrix(figure,16,1),drawlab=FALSE,diaglab=FALSE)
  dev.off()
  figure<-matrix(figure,4,4)
  png(paste0("../figures/2d_random_supervised_learning_",i,".png"),width=400, height=400)
  plot.sociomatrix(figure,drawlab=FALSE,diaglab=FALSE)
  dev.off()
  
}


###fixed patterns
for(i in 1:6){
  
  figure<-rep(0,16)
  figure[1+i]<-1
  figure[2+i]<-1
  figure[5+i]<-1
  figure[sample(length(figure),1)]<-1
  
  
  
  png(paste0("../figures/1d_pattern_supervised_learning_",i,".png"),width=150, height=400)
  plot.sociomatrix(matrix(figure,16,1),drawlab=FALSE,diaglab=FALSE)
  dev.off()
  figure<-matrix(figure,4,4)
  png(paste0("../figures/2d_pattern_supervised_learning_",i,".png"),width=400, height=400)
  plot.sociomatrix(figure,drawlab=FALSE,diaglab=FALSE)
  dev.off()
  
}

######print clustering pictures
x<-c(1:900)+0.1
y<-x
mult<-1200
y[1:300]<-mult*runif(300)
y[301:600]<-2000+mult*runif(300)
y[601:900]<-mult*runif(300)

points<-tibble(x,y)

subsample<-sample(length(x),400)
points_green<-points[subsample,]

ggplot(points,aes(x=x,y=y)) + geom_point() +
  geom_point(data=points_green,aes(x=x,y=y),col="green")
ggsave("../figures/clustering.png")  







