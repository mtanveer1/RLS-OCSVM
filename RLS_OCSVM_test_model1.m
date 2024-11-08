function [predicted_label, theta] = RLS_OCSVM_test_model1(data,tr_data,alpha,rho,FunPara)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
K=kernelfun(data,kerfPara,tr_data);
[samples, features]=size(K);
E=ones(samples,features);
K=K+(1/lambda)*E;
mat2=(-1/sqrt(lambda*gamma))*E;
K2=[K mat2];
f=K2*alpha-rho;
d=abs(f)/norm(alpha);
theta=max(d);
predicted_label=[];
for i=1:size(f,1)
    if d(i)<=theta
        h=1;
    else
        h=-1;
    end
    predicted_label=[predicted_label; h];
    time=toc;
end
end
%