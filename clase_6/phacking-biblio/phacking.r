## Ejemplo 1
rm(list = ls())
nreg <- 5000
vresp <- rnorm(nreg, 0, 1)
hist(vresp)

nsamp = 100
ssamp = 50
vc <- sample(vresp, ssamp, replace = FALSE)
vresp.df <- data.frame(vc)
for(i in 2:nsamp){
  vc <- sample(vresp, ssamp, replace = FALSE)
  vresp.df <- cbind(vresp.df, vc)
}
names(vresp.df) <- paste0("muestra", 1:100)
str(vresp.df)

p <- matrix(NA,nsamp,nsamp)
for(i in 1:(nsamp-1)){
  for(j in (i+1):nsamp){
    p[i,j]<-t.test(vresp.df[,i],vresp.df[,j])$p.value
  }
}
hist(as.vector(p),seq(0,1,by=0.025))

image(1:nsamp,1:nsamp,p,c(0,1),col=gray.colors(20))
#image(1:nsamp,1:nsamp,p<.05,c(0,1),col=gray.colors(20))
#image(1:nsamp,1:nsamp,p>.05 & p<.075,c(0,1),col=gray.colors(20))

ind <- which(p>0.05 & p<0.075,arr.ind = T)
i <- sample.int(dim(ind)[1],1) 
i1 = ind[i,1]
i2 = ind[i,2]
p[i1,i2]
t.test(vresp.df[,i1],vresp.df[,i2])$p.value

p2 <- matrix(NA,nsamp,1)
for(i in 1:nsamp){
  p2[i]<-t.test(c(vresp.df[,i1],sample(vresp, 5, replace = FALSE)),
                c(vresp.df[,i2],sample(vresp, 5, replace = FALSE)))$p.value
}
100*sum(p2<0.05)/length(p2)

p2 <- matrix(NA,nsamp,1)
for(i in 1:nsamp){
  p2[i]<-t.test(c(vresp.df[,i1],sample(vresp, 5, replace = FALSE)),
                c(vresp.df[,i2],sample(vresp, 5, replace = FALSE)))$p.value
}
100*sum(p2<0.05)/length(p2)

p2 <- matrix(NA,nsamp,1)
for(i in 1:nsamp){
  p2[i]<-t.test(c(vresp.df[,i1],sample(vresp, 10, replace = FALSE)),
         c(vresp.df[,i2],sample(vresp, 10, replace = FALSE)))$p.value
}
100*sum(p2<0.05)/length(p2)




