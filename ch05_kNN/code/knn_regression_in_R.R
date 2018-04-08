###################################
## k-nearest neighber regression ##
###################################


##################################
## 1. 환경설정 및 자료 불러오기 ##
##################################
rm(list=ls(all=TRUE)) # 기존 변수 제거
setwd("I:/CLASS/ML_DEEP_IMJ/Lab/Week6_lab5_kNN/code") # 코드가 있는 경로로 설정
# 재현성을 위한 seed 설정 
set.seed(2824)

# 데이터 불러오기
raw_cali_dat <- read.csv("../data/GOCI_ocean_cali.csv")
raw_vali_dat <- read.csv("../data/GOCI_ocean_vali.csv")
# 변수 선정
fselect_cali_dat <- raw_cali_dat[, c("SST", "SSS", "fCO2_SEA")]
fselect_vali_dat <- raw_vali_dat[, c("SST", "SSS", "fCO2_SEA")]
# 빠른 작업을 위해 두 클래스만 추출
#cali_dat = fselect_cali_dat[133:493,]
#vali_dat = fselect_vali_dat[33:121,]
cali_dat = fselect_cali_dat
vali_dat = fselect_vali_dat


##############################
## 2. kNN 분류 - k=1에 대해 ##
##############################
install.packages("FNN")
library(FNN)
var_num <- length(cali_dat)
train_x <- cali_dat[,-var_num]
test_x <- vali_dat[,-var_num]
train_y <- cali_dat[,var_num]
test_y <- vali_dat[,var_num]

knn_results <- knn.reg(train = train_x,
             test = test_x,
             y = train_y,
             k = 1,
             algorithm = "kd_tree")
#algorithm - algorithm=c("kd_tree", "cover_tree", "brute"))


# 수치적 정확도
RMSE <- sqrt(mean((knn_results$pred-test_y)^2))
paste("RMSE : ", RMSE)


###############################
## 4. kNN 분류 - 최적 k 찾기 ##
###############################
accum_k <- NULL # 각 k에 대한 정확도 저장용
# kk가 1에서 총 샘플 개수까지 정확도 변화 테스트
# kk in c(1:nrow(train_x))
for(kk in c(1:20)){
  knn_results <- knn.reg(train = train_x,
                     test = test_x,
                     y = train_y,
                     k = kk,
                     algorithm = "kd_tree")
  RMSE <- sqrt(mean((knn_results$pred-test_y)^2))
  accum_k <- c(accum_k, RMSE)
}
# 결과물을 table로
kk_test <- data.frame(k = c(1:20), RMSE = accum_k)
x11()
plot(formula = RMSE ~ k,
     data = kk_test,
     type = "o",
     pch = 20,
     main = "Test - Optimal K")
with(kk_test, text(RMSE ~ k, labels = rownames(kk_test),pos = 1, cex = 0.7))
best_k = min(kk_test[kk_test$RMSE %in% min(kk_test$RMSE), "k"])
paste("The best k is ", best_k)

  
  
  
#####################################
## 5. kNN 분류 - 최적 k로 분류평가 ##
#####################################
knn_results <- knn.reg(train = train_x,
                       test = test_x,
                       y = train_y,
                       k = best_k,
                       algorithm = "kd_tree")
RMSE <- sqrt(mean((knn_results$pred-test_y)^2))
paste("RMSE : ", RMSE)


