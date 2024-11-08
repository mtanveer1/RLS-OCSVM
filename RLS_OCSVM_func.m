function [alpha, rho, train_time] = RLS_OCSVM_func(train_data,~,FunPara)
N=size(train_data,1);
gamma=FunPara.c_1;
kerfPara = FunPara.kerfpara;
K=kernelfun(train_data,kerfPara,train_data);
[samples,~]=size(K);
lambda=FunPara.c_2;
E=ones(samples,samples);
K2=[K+(1/lambda)*E (-1/sqrt(lambda*gamma))*E; (-1/sqrt(lambda*gamma))*E (1/gamma)*E];
e1=ones(samples,1);
z=zeros(samples,1);
a_tilde=[e1; z];

I2=eye(2*samples);
R=(1+(sqrt(gamma)/2)); M=(1+sqrt(lambda/gamma));
RM=R/M; e2=[M*ones(samples,1); z];
tic;
mat1=((I2/gamma)*N+K2);
deno=a_tilde'*(mat1\e2);
alpha=(mat1\e2)/deno;
rho=RM/(deno);
train_time=toc;

end