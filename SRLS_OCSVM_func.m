function [BS,xB,b, train_time,base_time]=SRLS_OCSVM_func(train, ker, subsetsize,errorbound,gamma,lambda,bflag,seed)  
%%Solve the P-LSSVM by pivoted Cholesky decomposing kernel matrix, 
%% C1 is gamma   C2 is lambda
%Output:
rng(seed);
m=size(train,1);   r=1;
BS=zeros(0,subsetsize);
N= 1:m;
d_K=ones(m,1);      
I=randperm(m);d_K(I(1))=d_K(I(1))+1;%%make the first random selection.
error(r) = sum(d_K);
P=zeros(m,0);
tic
while error(r)>errorbound && r<=subsetsize
    [~,index]=max(d_K(N));s_in=N(index);N(index)=[];
    k_in=kernelfun(train,ker,train(s_in,:)); 
    if r==1
        p=k_in/sqrt(k_in(s_in));
    else
        u=P(s_in,:)';nu=sqrt(k_in(s_in)-u'*u);
        p=(k_in-P*u)/nu;
        p(BS)=0;
    end
    P(:,r)=p; BS(r)=s_in;
    d_K(N)=d_K(N)-p(N).^2; 
    error(r+1) = sum(d_K(N));
    r=r+1;
end
base_time=toc;
tic
if bflag==1 %%output  the solutions
    K_MB=P*P(BS,:);
    K_BB=P(BS,:);
    ee=-1/sqrt(lambda*gamma)*ones(size(K_BB,1),1);
    ee2=-1/sqrt(lambda*gamma)*ones(size(K_MB,1),1);
    K_BB_tilde=[K_BB+(1/lambda)*ones(size(K_BB))  ee; ee' (1/gamma)];
    K_MB_tilde=[K_MB+(1/lambda)*ones(size(K_MB)) ee2; ee2(1:size(K_MB,2))' (1/gamma)];
    R=(1+(sqrt(gamma)/2)); M=(1+sqrt(lambda/gamma)); e2=[M*ones(m,1); 0];
    e22=e2'*e2;
    Sum=e2'*K_MB_tilde;
    xB_deno=K_BB_tilde+(gamma/m)*(K_MB_tilde'*K_MB_tilde)-(gamma/(2*m^2))*(Sum'*Sum)/(e22);
    xB_deno=xB_deno+10^-4*eye(size(xB_deno));
    det(xB_deno)
    xB=xB_deno\((R/(2*m*e22))*Sum');
    b=(1/2*gamma*e22)*(R+(gamma/m)*xB'*Sum');
end
train_time=toc;
return

