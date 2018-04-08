
training<- read.csv(file="Warning_train.csv",head=TRUE)
View(training)
training$Warning <-factor(training$Warning)
training$Warning <-relevel(training$Warning, ref="0")
# 0은 기준이 되는것 즉, 많은 것ㄷ

library(nnet)
model <- multinom(Warning~., data=training)
summary(model)
predict(model,training)
predict(model,, type="prob")
cm <- table(predict(model,training), training$Warning)
print(cm)


testing<- read.csv(file="Warning_test.csv",head=TRUE)
testing$Warning <-factor(testing$Warning)
testing$Warning <-relevel(testing$Warning, ref="0")
predict(model,testing)

cm <- table(predict(model,testing), testing$Warning)
print(cm)
sum(diag(cm))/sum(cm)
