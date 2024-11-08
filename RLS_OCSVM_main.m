clc; clear all; close all;
%% define hyperparameter range
C1=[10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3]; %C1=gamma
C2=[10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3]; %C2=lambda
Sigma = [2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
FunPara.kerfpara.type='rbf'; seed=[1 2 3 4 5];
%% load data
 load('train_data.mat');
 load('validation_data.mat');
 load('test_data.mat');

%% Define a cell array and store each structure in a separate cell
train_data_cells = {train_data.train_data_run1, train_data.train_data_run2, train_data.train_data_run3, train_data.train_data_run4, train_data.train_data_run5};
Test_data_cells = {test_data.test_data_run1, test_data.test_data_run2, test_data.test_data_run3, test_data.test_data_run4, test_data.test_data_run5};
validation_data_cells = {validation_data.validation_data_run1, validation_data.validation_data_run2, validation_data.validation_data_run3, validation_data.validation_data_run4, validation_data.validation_data_run5};

tot_run = 5;
for run=1:tot_run    
    run   
    Train_data=train_data_cells{run};   
    Validation_data=validation_data_cells{run};   
    Test_data=Test_data_cells{run};  

    %% tuning parameters
      max_gm=0;
    for ii = 1:length(C1)
        FunPara.c_1=C1(ii);
        for iii = 1:length(C2)
            FunPara.c_2=C2(iii);
            for t = 1:length(Sigma)
                FunPara.kerfpara.pars=Sigma(t);
                FunPara.kerfpara.type='rbf';
                traindata = Train_data(:,1:end-1);
                valdata = Validation_data(:,1:end-1);
                vallabel = Validation_data(:,end);
                
                %% training
                [alpha,rho,~]=RLS_OCSVM_func(traindata,traindata,FunPara);
                [labeltr_OCLSSVM,theta_train] = RLS_OCSVM_test_model1(traindata,traindata,alpha,rho,FunPara);
                labelval_OCLSSVM = RLS_OCSVM_test_model2(valdata,traindata,alpha,rho,FunPara,theta_train);
                gm = Evaluate(vallabel,labelval_OCLSSVM,1);

                if gm>max_gm
                    max_gm=gm;
                    Funpara.c_1=FunPara.c_1;
                    Funpara.c_2=FunPara.c_2;
                    Funpara.kerfpara.pars=FunPara.kerfpara.pars;
                    Funpara.kerfpara.type='rbf';
                end
            end
        end
    end
    traindata=[Train_data(:,1:end-1); Validation_data(:,1:end-1)];
    testdata=Test_data(:,1:end-1);
    testlabel=Test_data(:,end);
    %% testing
    [alpha,rho,train_time]=RLS_OCSVM_func(traindata,traindata,Funpara);
    label_OCLSSVM = RLS_OCSVM_test_model3(testdata,traindata,alpha,rho,Funpara);
    gmean = Evaluate(testlabel,label_OCLSSVM,1);
    mgmean(run) = gmean;
    clear traindata trainlabel testdata testlabel label_OCLSSVM  gmean
end
mmtest_mean=mean(mgmean)




