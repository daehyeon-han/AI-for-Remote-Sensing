% 'BoxConstraint'(상자제약조건) : 마진을 위반하는 관측값에 적용되는 최대 벌점을 제어하고 과잉피팅을 방지(정규화)하는
%                               데 도움이 되는 모수로 상자 제약 조건을 늘리면 SVM 분류기가 더 적은 서포트 벡터를 할당합니다. 
%                               그러나, 상자 제약 조건을 늘리면 훈련 시간이 더 길어질 수 있습니다.(디폴트
%                               : 1)
% 'KernelFunction'(커널함수) : 'KernelFunction'과 함께 커널 함수 이름이 쉼표로 구분되어 지정됩니다.
%                              'gaussian' 또는 'rbf' - 가우스 커널 또는 방사 기저 함수(RBF) 커널로, 단일 클래스 학습에 대한 디폴트 값임
%                              'linear' -  선형 커널로, 2-클래스 학습에 대한 디폴트 값임 
%                              'polynomial' - 다항식 커널. 'PolynomialOrder',q를 사용하여 차수가 q인 다항식 커널을 지정%
% 'PolynomialOrder' : 다항식 커널 함수 차수로, 'PolynomialOrder'와 함께 양의 정수가 쉼표로 구분되어
%                     지정됩니다.(디폴트 : 3)
% 'KernelScale'(커널 스케일) : 'KernelScale'과 함께 'auto' 또는 양의 스칼라가 쉼표로 구분되어 지정됩니다. 
%                              소프트웨어는 예측 변수 행렬 X의 모든 요소를 KernelScale의 값으로 나눕니다. 
%                              그런 다음, 적합한 커널 노름(Norm)을 적용하여 그람 행렬을
%                              계산합니다.(디폴트 :1)
% 'Standardize'(예측 변수 데이터를 표준화) : 예측 변수 데이터를 표준화하는 플래그로, 'Standardize'와 함께 true(1) 또는 false(0)이 쉼표로 구분되어 지정됩니다.

% 'OptimizeHyperparameters'(최적화할 모수) : 'none'(최적화안함)/
%                                           'auto'(BoxConstraint,KernelScale 만 조정)
%                                           'all'(모두)
% BoxConstraint : fitcsvm이 기본적으로 범위 [1e-3,1e3]에서 로그 스케일링된 양수 값 중에서 검색을 수행합니다.
% KernelScale : fitcsvm이 기본적으로 범위 [1e-3,1e3]에서 로그 스케일링된 양수 값 중에서 검색을 수행합니다.
% KernelFunction : fitcsvm이 'gaussian', 'linear', 'polynomial' 중에서 검색을 수행합니다.
% PolynomialOrder : fitcsvm이 범위 [2,4] 내 정수 중에서 검색을 수행합니다.
% Standardize : fitcsvm이 'true' 및 'false' 중에서 검색을 수행합니다.
%
% 'IterationLimit'(수치최적화 반복의 최대 횟수) : 수치 최적화 반복의 최대 횟수로, 'IterationLimit'와 함께 양의 정수가 쉼표로 구분되어 지정됩니다.
%                                               최적화 루틴이 성공적으로 수렴되는지 여부에 상관없이 소프트웨어가 훈련된 모델을 반환합니다.
%                                               (디폴트 : 1e8)
% 'HyperparameterOptimizationOptions' : 참고)https://kr.mathworks.com/help/stats/fitcsvm.html?s_tid=doc_ta#bt9w6j6-OptimizeHyperparameters
% Grid search 에서 시간 많이 걸리는 거 주의!!!
%% classification
clc;clear;
path = 'D:\class_SVM\matlab\';

% open calibration file as table
cali = readtable('climate_zone_cali.csv');
cali_input = table2array(cali(:,1:size(cali,2)-1)); % calibration input
cali_target = table2array(cali(:,12)); %calibration target

% open validation file as table
vali = readtable'climate_zone_vali.csv']);
vali_input = table2array(vali(:,1:size(cali,2)-1)); % validation input
vali_target = table2array(vali(:,12)); %validation target

classes = unique(cali_target); % class type
SVMModels = cell(size(classes,1),1); 

for j = 1:numel(classes);
    % Create binary classes for each classifier
    indx = strcmp(cali_target,classes(j)); 
    % Train an SVM classifier using cali_input data and indx and Store the classifier in a cell of a cell array.
%     SVMModels{j} = fitcsvm(cali_input,indx,'ClassNames',[false true],'Standardize',true,...
%          'KernelFunction','rbf','BoxConstraint',1); 
%    optimization   
   SVMModels{j} = fitcsvm(cali_input,indx,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch','MaxObjectiveEvaluations',5),'IterationLimit',5)
end

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},vali_input);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
[~,maxScore] = max(Scores,[],2);

% if you want to get validation results as label name,
% for i = 1:size(maxScore,1); 
%     output{i,1} = classes{maxScore(i,1),1};
% end


%% regression
clc;clear;
path = 'G:\class_SVM\matlab\';
cali = readtable([path 'PM10_cal_AIRS.csv']); % open calibration file
vali = readtable([path 'PM10_val_AIRS.csv']); %v open validation file

SVRmodel = fitrsvm(cali,'PM10','KernelFunction','gaussian','KernelScale','auto','Standardize',true);
% optimization
%SVRmodel = fitrsvm(cali,'PM10','OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch'));
%'IterationLimit',5

% sturct 구조 안에서 parameter 수정?

test = predict(SVRmodel,vali);
scatter(table2array(vali(:,11)),test)

