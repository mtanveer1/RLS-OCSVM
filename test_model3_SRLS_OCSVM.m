function [predicted_label, time] = test_model3_SRLS_OCSVM(data,tr_data,alpha,rho,FunPara,PCP)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
K=kernelfun(data,kerfPara,tr_data(PCP,:));
[samples, features]=size(K);
E=ones(samples,features);
K=K+(1/lambda)*E;
mat1=(-1/sqrt(lambda*gamma))*E(:,end);
KK=[K mat1];

K2=kernelfun(tr_data,kerfPara,tr_data(PCP,:));
E2=ones(size(K2));
K2=K2+(1/lambda)*E2;
mat2=(-1/sqrt(lambda*gamma))*E2(:,end);
KK2=[K2 mat2];
f=KK2*alpha-rho;
d2=abs(f)/norm(alpha);
theta_train=max(d2);
predicted_label=[];
f=KK*alpha-rho;
d=abs(f)/norm(alpha);
% S=mean(d);
predicted_label=[];
for i=1:size(f,1)
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



