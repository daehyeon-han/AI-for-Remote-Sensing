setwd("/lab/data")
# https://github.com/mananshah99/rotationforest
## 1. Install the rotationForest package from github ------------------
install.packages('devtools') # Only if needed
require('devtools')
devtools::install_github('mananshah99/rotationforest')
require('rotationForest')

## 2. Install other librarys ------------------------------------------
install.packages('Metrics')
install.packages('caret')

library(Metrics)
library(caret)

## 3. Read calibration files ------------------------------------------
# classification
data <- read.table("classification_tr.csv", sep = ",", header = TRUE)
names(data)
n <- ncol(data)
data.dependent <- data[,1:(n-1)]
data.response <- data[,n]
data.response <- as.factor(data.response) # for calibration

# regression - X
#data <- read.table("GOCI_ocean_cali.csv", sep = ",", header = TRUE)
#names(data)
#n <- ncol(data)
#data.dependent <- data[,1:(n-1)]
#data.response <- data[,n]

## 4. rotation forest: rotationForest(x, y, K, L) ---------------------
rotF <- rotationForest(data.dependent, data.response, 3, 10, verbose = FALSE)
cali_predict <- predict(rotF, data.dependent, prob = FALSE)

## 5. calibration accuracy --------------------------------------------
mean(data.response==cali_predict)
# accuracy assessment
confusionMatrix(data.response, cali_predict)

## 6. Validation ------------------------------------------------------
valid <- read.table("classification_va.csv", sep = ",", header = TRUE)
n_vali <- ncol(valid)
x_vali <- valid[,1:(n-1)]
y_vali <- valid[,n]
y_vali <- as.factor(y_vali) # for calibration

vali_predict <- predict(rotF,x_vali, prob = FALSE)
# write.table(yhat_vali, "GOCI_ocean_vali_result.csv", sep=",", append=FALSE)

## 7. validation accuracy --------------------------------------------
mean(y_vali==vali_predict)
# accuracy assessment
confusionMatrix(y_vali, vali_predict)
