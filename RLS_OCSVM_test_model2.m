function [predicted_label, time] = RLS_OCSVM_test_model2(data,tr_data,alpha,rho,FunPara,theta_train)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
predicted_label=[];
K=kernelfun(data,kerfPara,tr_data);
[samples, features]=size(K);
E=ones(samples,features);
K=K+(1/lambda)*E;
mat2=(-1/sqrt(lambda*gamma))*E;
K2=[K mat2];
f=K2*alpha-rho;
d=abs(f)/norm(alpha);
% S=mean(d);
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

% predicted_label=sign(K2*alpha-rho);
time=toc;
end

%