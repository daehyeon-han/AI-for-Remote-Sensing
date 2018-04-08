## Installing (Java?? R?? ?????? ??????)

system('defaults write org.R-project.R force.LANG en_US.UTF-8')

install.packages('extraTrees')
install.packages('Metrics')
install.packages('caret')
install.packages('e1071')
install.packages('lattice')
install.packages('ggplot2')

## Loading ExtraTrees
library(extraTrees)
library(Metrics)
library(caret)

## Read the data file
setwd('/Users/dhan/Dropbox/Archive/_coursework/2018_1st/AI_RS/week3/Week3_lab2/data')
calib <- read.csv(file="classification_tr.csv")
names(calib)
n <- ncol(calib)
x <- calib[,1:(n-1)]
y <- calib[,n]

## Regression with ExtraTrees
yhat <- predict(et,x)
et <- extraTrees(x, y, ntree=500, mtry=5, nodesize=5, numRandomCuts=2)
rmse(y,yhat)

## Classification with ExtraTrees
y<-as.factor(y) # classification
et <- extraTrees(x, y, ntree=500, mtry=5, nodesize=5, numRandomCuts=2)
yhat <- predict(et,newdata=x)
# accuracy
mean(y==yhat)
# accuracy assessment
confusionMatrix(yhat, y)

## Save results
write.table(yhat, "GOCI_ocean_cali_result.csv", sep=",", append=FALSE)
save(et, file="Extra_Trees.RData")

## Validation
valid <- read.csv(file="GOCI_ocean_vali.csv")
x_vali <- valid[,1:(n-1)]
y_vali <- valid[,n]
yhat_vali <- predict(et,newdata=x_vali)
write.table(yhat_vali, "GOCI_ocean_vali_result.csv", sep=",", append=FALSE)