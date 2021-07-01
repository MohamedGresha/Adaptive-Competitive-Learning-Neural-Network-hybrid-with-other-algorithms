function [bestk,bestw,ACL,clusterid,NMI,NN]=demo_p_1()
% this is demo1.m a function tests differnet algorithms in estimating the number
% of clusters of the Bank Marketing data set.
%
%NMI is the normalized mutual information between the cluster id vector and the class id vector. 

%Ahmed Rafat
%Mohamed Gresha
%Oct. 18

load ('data_1.mat');

[X,varmin,varrange]=atscale(X);
y=X';

%% Select the algorithm
%[bestk,bestmu,bestcov,bestpp,clusterid] = mixtures4(y);         %Figueiredo 2002********
%[bestk,bestmu,bestcov,bestpp,clusterid]=atembic(y');            %BIC
%[bestk,bestmu,bestcov,bestpp,clusterid]=atemmi(y');             %MI***********
%[bestk,bestmu,bestcov,bestpp,clusterid]=atemmi_m(y');           %MI+CEM_Modified******
%[bestk,bestmu,bestcov,bestpp,MIPL,clusterid]=atemmipl(y');      %MI+CEM_PL******
[bestk,bestw,ACL,clusterid,NN]=atacl_parfor(y');      % Competitive neural network

% compute the normalized mutual information
if bestk == 1
    NMI = 0;
else
    classid=[1*ones(5000,1);
         2*ones(5000,1);
         3*ones(5000,1);
         4*ones(5000,1);
         5*ones(5000,1);
         6*ones(5000,1);
         7*ones(5000,1);
         8*ones(5000,1);
         9*ones(5000,1);
         10*ones(5000,1);
         11*ones(5000,1);
         12*ones(5000,1);
         13*ones(5000,1);
         14*ones(5000,1);
         15*ones(5000,1);
         16*ones(5000,1);
         17*ones(5000,1);
         18*ones(5000,1);
         19*ones(5000,1);
         20*ones(5000,1)];
    pclass = [5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000 5000/100000];   %the probability of each class
    pcluster=[];                %bestpp;        %the probability of each cluster
    pclass_cluster=[];      %the probability that a member of cluster j belongs to class i
    n = length(classid);
    for i=1:20
        if i==1
            c1=1;
            c2=5000;
        elseif i==2
            c1=5001;
            c2=10000;
        elseif i==3
            c1=10001;
            c2=15000;
        elseif i==4
            c1=15001;
            c2=20000;
        elseif i==5
            c1=20001;
            c2=25000;
        elseif i==6
            c1=25001;
            c2=30000;
        elseif i==7
            c1=30001;
            c2=35000;
        elseif i==8
            c1=35001;
            c2=40000;
        elseif i==9
            c1=40001;
            c2=45000;
        elseif i==10
            c1=45001;
            c2=50000;
        elseif i==11
            c1=50001;
            c2=55000;
        elseif i==12
            c1=55001;
            c2=60000;
        elseif i==13
            c1=60001;
            c2=65000;
        elseif i==14
            c1=65001;
            c2=70000;
        elseif i==15
            c1=70001;
            c2=75000;
        elseif i==16
            c1=75001;
            c2=80000;
        elseif i==17
            c1=80001;
            c2=85001;
        elseif i==18
            c1=85001;
            c2=90000;
        elseif i==19
            c1=90001;
            c2=95000;
        elseif i==20
            c1=95001;
            c2=100000;
        end
        for j=1:bestk
            nj=length(find(clusterid == j));
            nij=length(find(clusterid(c1:c2) == j));
            pclass_cluster(i,j)=nij/n;
            pcluster(j)= nj/n;
        end
    end
    [g,h]=size(pcluster);
    for w=1:h
        if pcluster(w)== 0
            pcluster(w)= 0.000001;
        end
    end
    Hclass = -sum(pclass.*log(pclass)/log(2));
    Hcluster = -sum(pcluster.*log(pcluster)/log(2));

    MI=0;       % the Mutual information
    for i=1:20
        for j=1:bestk
            if pclass_cluster(i,j) ~= 0
                MI = MI + pclass_cluster(i,j) * log(pclass_cluster(i,j)/(pclass(i) * pcluster(j)))/log(2);
            end
        end
    end
    NMI = MI / sqrt(Hclass * Hcluster);
end
return;        