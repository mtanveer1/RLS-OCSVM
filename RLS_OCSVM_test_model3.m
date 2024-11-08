function [predicted_label, time] = RLS_OCSVM_test_model3(data,tr_data,alpha,rho,FunPara)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
K=kernelfun(data,kerfPara,tr_data);
[samples, features]=size(K);
E=ones(samples,features);

K=K+(1/lambda)*E;
mat1=(-1/sqrt(lambda*gamma))*E;
KK=[K mat1];

K2=kernelfun(tr_data,kerfPara,tr_data);
E2=ones(size(K2));
K2=K2+(1/lambda)*E2;
mat2=(-1/sqrt(lambda*gamma))*E2;
KK2=[K2 mat2];
f=KK2*alpha-rho;
d2=abs(f)/norm(alpha);
theta_train=max(d2);
predicted_label=[];
f=KK*alpha-rho;
d=abs(f)/norm(alpha);

for i=1:size(f,1)
if d(i)<=theta_train
    h=1;
else
    h=-1;
end
predicted_label=[predicted_label; h];
% predicted_label=sign(K*alpha-rho);
end
time=toc;
end
% simple LSSVM
