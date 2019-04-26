
"../figures/"


library(sna)
library(ggplot2)


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




ggsave()

as.vector(figure)
