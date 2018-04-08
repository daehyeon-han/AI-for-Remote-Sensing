
install.packages("e1071")
install.packages('kernlab')

library(e1071)
library(kernlab)

!setwd("C:/Users/Seohui Park/Desktop/SVM_AIRS_lab/")

cal_class = read.csv(file="기후대분류/calimate_zone_cali.csv")
val_class = read.csv(file="기후대분류/calimate_zone_vali.csv")


## classification mode
# default with factor response:
model <- svm(target ~ ., data = cal_class)
summary(model)


## alternatively the traditional interface:
#head(cal_class,10)
#attach(cal_class) # 데이터를 R 검색 경로에 추가하여 변수명으로 접근가능하게 함. 해제는 detach
#x <- subset(cal_class, select = -target)
#y <- target
#model_2 <- svm(x, y) 
#summary(model_2)


# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)
system.time(pred <- predict(model,x))

# Check accuracy:
y <- cal_class$target
table(pred, y)

attach(val_class) # 데이터를 R 검색 경로에 추가하여 변수명으로 접근가능하게 함. 해제는 detach
x_val <- subset(val_class, select = -target)
y_val <- target

pred_val <- predict(model, x_val)
table(pred_val, y_val)

# Tuning SVM to find best cost and gamma
svm_tune <- tune(svm, train.x=x, train.y=y, kernel="radial", ranges=list(cost=10^(-1:3), gamma=2^(-5:1))) #, gamma=c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.5,2)
print(svm_tune)

#After finding the best cost and gamma
svm_model_after_tune <- svm(target ~ ., data=cal_class, kernel="radial", cost=1000, gamma=0.025) #scale=TRUE/ kernel=linear,polynomial, radial, sigmoid
summary(svm_model_after_tune)


pred_val_after_tune <- predict(svm_model_after_tune, x_val)
table(pred_val_after_tune, y_val)

# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(cal_class[,-11])),
     col = as.integer(cal_class[,11]),
     pch = c("o","+")[1:1247 %in% svm_model_after_tune$index + 1])




## try regression mode 

install.packages("e1071")
install.packages('kernlab')

library(e1071)

setwd("C:/Users/Seohui Park/Desktop/SVM_AIRS_lab/")

cal_reg = read.csv(file="미세먼지추정/PM10_cal_AIRS.csv")
val_reg = read.csv(file="미세먼지추정/PM10_val_AIRS.csv")

head(cal_reg,10)

attach(cal_reg)
x <- subset(cal_reg, select = -PM10)
y <- PM10

# estimate model and predict input values
model   <- svm(x, y)
predicted_Y <- predict(model, x)

# visualize
plot(y,predicted_Y)

# test:
newdata <- data.frame(val_reg[,-11])
predicted_Y_val = predict(model, newdata)

x_val<-val_reg[,-11]
y_val<-val_reg[,11]

# visualize
plot(y_val,predicted_Y_val)


## Tuning SVR model by varying values of maximum allowable error and cost parameter
#Tune the SVM model
OptModel=tune(svm, PM10~., data=cal_reg,ranges=list(elsilon=seq(0,1), cost=10^(-1:3)))
model_tune<-OptModel$best.model
predicted_Y_tune <- predict(model_tune, x)
predicted_Y_val_tune <- predict(model_tune, x_val)

# visualize
plot(y_val,predicted_Y_val)
points(y_val, predicted_Y_val_tune, col = 4)

