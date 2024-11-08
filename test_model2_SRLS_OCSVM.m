function [predicted_label, time] = test_model2_SRLS_OCSVM(data,tr_data,alpha,rho,FunPara,theta_train,PCP)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
K=kernelfun(data,kerfPara,tr_data(PCP,:));
[samples, features]=size(K);
E=ones(samples,features);
K=K+(1/lambda)*E;
mat2=(-1/sqrt(lambda*gamma))*E(:,1);
K2=[K mat2];
predicted_label=[];
f=K2*alpha-rho;
d=abs(f)/norm(alpha);
for i=1:size(f,1)
% S=mean(d);
if d(i)<=theta_train
    h=1;
else
    h=-1;
end
predicted_label=[predicted_label; h];
time=toc;
end
end
%%



