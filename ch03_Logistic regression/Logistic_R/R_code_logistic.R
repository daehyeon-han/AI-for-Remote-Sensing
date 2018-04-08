training<- read.csv(file="Heatwave_train.csv",head=TRUE)
testing<- read.csv(file="Heatwave_test.csv",head=TRUE)

View(training)
View(testing)

model <-glm(Heatwave~. , training, family= "binomial")
summary(model)

res<- predict(model, testing, type="response")
res
testing

table(predictedvalue=res>0.5, Actualvalue=testing$Heatwave)


model <-glm(Heatwave~. -Tmin_b -Rhmin -Rhmax -Tmin, training, family= "binomial")
summary(model)
res<- predict(model, testing, type="response")
table(predictedvalue=res>0.5, Actualvalue=testing$Heatwave)



model <-glm(Heatwave~. , training, family= "binomial")  
#Heatwave~. , training은 training 안에 heatwave 빼고 다 입력값으로
#model <-glm(Heatwave~. -Tmin_b,-Rhmin , training, family= "binomial")  
#-변수 빼고 돌린ㄷ

res<- predict(model, training, type="response")
table(predictedvalue=res>0.5, Actualvalue=training$Heatwave)


#threshold test @ ROC curve
install.packages("ROCR")
library("ROCR")
ROCRPred<-prediction(res,training$Heatwave)
ROCRPref <-performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

res<- predict(model, training, type="response")
table(predictedvalue=res>0.5, Actualvalue=training$Heatwave)
table(predictedvalue=res>0.45, Actualvalue=training$Heatwave)
table(predictedvalue=res>0.42, Actualvalue=training$Heatwave)
table(predictedvalue=res>0.4, Actualvalue=training$Heatwave)


res<- predict(model, testing, type="response")
table(predictedvalue=res>0.5, Actualvalue=testing$Heatwave)
table(predictedvalue=res>0.45, Actualvalue=testing$Heatwave)
table(predictedvalue=res>0.42, Actualvalue=testing$Heatwave)
table(predictedvalue=res>0.4, Actualvalue=testing$Heatwave)


