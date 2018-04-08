###################################
## k-nearest neighber clustering ##
###################################


##################################
## 1. 환경설정 및 자료 불러오기 ##
##################################
rm(list=ls(all=TRUE)) # 기존 변수 제거
setwd("I:/CLASS/ML_DEEP_IMJ/Lab/Week6_lab5_kNN/code") # 코드가 있는 경로로 설정
# 재현성을 위한 seed 설정 
set.seed(2824)

# 데이터 불러오기
raw_cali_dat <- read.csv("../data/classification_tr.csv")
raw_vali_dat <- read.csv("../data/classification_va.csv")

# 변수 선정
fselect_cali_dat <- raw_cali_dat[, c("spr_b3", "win_b3", "class")]
fselect_vali_dat <- raw_vali_dat[, c("spr_b3", "win_b3", "class")]

# 빠른 작업을 위해 두 클래스만 추출
cali_dat <- fselect_cali_dat[133:493,]
vali_dat <- fselect_vali_dat[33:121,]

# Min max추출
dummy01 <- rbind(cali_dat, vali_dat) #min, max를 뽑아내기 위해 합
max_of_all <- apply(dummy01, MARGIN = 2, function(x) max(x, na.rm=TRUE))
min_of_all <- apply(dummy01, MARGIN = 2, function(x) min(x, na.rm=TRUE))

# Cali normalize
matrix_max <- matrix(rep(max_of_all,each=nrow(cali_dat)),nrow=nrow(cali_dat)) # cali 파일과 같은 사이즈로 max, min 매트릭스 생성
matrix_min <- matrix(rep(min_of_all,each=nrow(cali_dat)),nrow=nrow(cali_dat))
ncali_dat <- (cali_dat - matrix_min)/ (matrix_max - matrix_min) # 앞에 붙은 n은 normalized의 줄임
ncali_dat$class <- cali_dat$class

# vali normalize
matrix_max <- matrix(rep(max_of_all,each=nrow(vali_dat)),nrow=nrow(vali_dat))
matrix_min <- matrix(rep(min_of_all,each=nrow(vali_dat)),nrow=nrow(vali_dat))
nvali_dat <- (vali_dat - matrix_min)/ (matrix_max - matrix_min)
nvali_dat$class <- vali_dat$class





#########################
## 2. 데이터 분포 확인 ##
#########################

install.packages("scales") 
library(scales)

x11() # 새 창
# train 산점도 그리기 
plot(formula = spr_b3 ~ win_b3, 
     data = ncali_dat, 
     col = alpha(c("blue", "green"), 1)[ncali_dat$class], 
     main = "train - Classification classes") 
# validation해야할 자료들 표시하기
points(formula = spr_b3 ~ win_b3, 
       data = nvali_dat, 
       pch = 17, 
       cex = 1.2, 
       col = "red") 
# 범례 그리기 
legend("topleft", c("class 1", "class 2", "vali_dat"), pch = c(1, 1, 17), col = c(alpha(c("blue", "green"), 1), "red"), cex = 0.9)





##############################
## 3. kNN 분류 - k=1에 대해 ##
##############################
install.packages("lcass")
library(class)
var_num <- length(cali_dat)
train_x <- ncali_dat[,-var_num]
test_x <- nvali_dat[,-var_num]
train_y <- ncali_dat[,var_num]
test_y <- nvali_dat[,var_num]

knn_results <- knn(train = train_x,
             test = test_x,
             cl = train_y,
             k = 1,
             prob = FALSE,
             use.all = FALSE)
#use all - controls handling of ties. If true, all distances equal to the kth largest are included.
#If false, a random selection of distances equal to the kth is chosen to use
#exactly k neighbours.

# 수치적 정확도
OA <- sum(knn_results == test_y) / length(test_y)*100 ; 
paste("Overall accuracy : ", OA, " %")




###############################
## 4. kNN 분류 - 최적 k 찾기 ##
###############################
accum_k <- NULL # 각 k에 대한 정확도 저장용
# kk가 1에서 총 샘플 개수까지 정확도 변화 테스트
# kk in c(1:nrow(train_x))
for(kk in c(1:20)){
  knn_results <- knn(train = train_x,
                     test = test_x,
                     cl = train_y,
                     k = kk)
  OA <- sum(knn_results == test_y) / length(test_y)*100 ; 
  accum_k <- c(accum_k, OA)
}
# 결과물을 table로
kk_test <- data.frame(k = c(1:20), OA = accum_k)
x11()
plot(formula = OA ~ k,
     data = kk_test,
     type = "o",
     pch = 20,
     main = "Test - Optimal K")
with(kk_test, text(OA ~ k, labels = rownames(kk_test),pos = 1, cex = 0.7))
best_k = min(kk_test[kk_test$OA %in% max(kk_test$OA), "k"])
paste("The best k is ", best_k)

  
  
  
#####################################
## 5. kNN 분류 - 최적 k로 분류평가 ##
#####################################
knn_results <- knn(train = train_x,
                   test = test_x,
                   cl = train_y,
                   k = best_k)
OA <- sum(knn_results == test_y) / length(test_y)*100 ; 

cross.tab <- table(knn_results, test_y)
cross.tab

paste("Overall accuracy : ", OA, " %")


