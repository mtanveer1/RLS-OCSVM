
function [predicted_label, theta] = test_model1_SRLS_OCSVM(data,tr_data,alpha,rho,FunPara,PCP)
kerfPara = FunPara.kerfpara;
gamma=FunPara.c_1;
lambda=FunPara.c_2;
tic;
K=kernelfun(data,kerfPara,tr_data(PCP,:));
E=ones(size(K));
K=K+(1/lambda)*E;
mat2=(-1/sqrt(lambda*gamma))*E(:,1);
K2=[K mat2];
predicted_label=[];
f=K2*alpha-rho;
d=abs(f)/norm(alpha);
theta=max(d);
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
%%



