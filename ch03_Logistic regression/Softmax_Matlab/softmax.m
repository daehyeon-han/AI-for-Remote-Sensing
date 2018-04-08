

clear; clear all; clc

%% trainning
data = xlsread('Warning_train.csv');
X = data(:, [1:6]); Y = data(:, 7);
sp = categorical(Y);
[B,dev,stats] = mnrfit(X,sp);
sp=double(sp); %이부분 주의, class 값이 index로 바뀐다.
% test = grp2idx(sp);

pihat=mnrval(B,X)
[A,get]=max(pihat,[],2)
confusionmat(get,sp)

%% testing
test = xlsread('Warning_test.csv');
X2 = test(:, [1:6]); Y2 = test(:, 7);
sp2 = categorical(Y2);
pihat2=mnrval(B,X2);
[A2,get2]=max(pihat2,[],2);
sp2=double(sp2); %이부분 주의, class 값이 index로 바뀐다.
confusionmat(get2,sp2)


% 변수 중요도 
