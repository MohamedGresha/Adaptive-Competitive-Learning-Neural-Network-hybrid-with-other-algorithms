function [bestk,bestw,ACL,clusterid,NMI,NN]=demo_p()
% this is demo1.m a function tests differnet algorithms in estimating the number
% of clusters of the Bank Marketing data set.
%
%NMI is the normalized mutual information between the cluster id vector and the class id vector. 

%Ahmed Rafat
%Mohamed Gresha
%Oct. 18

load ('data.mat');

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
    classid=[1*ones(25000,1);
         2*ones(25000,1);
         3*ones(25000,1);
         4*ones(25000,1);
         5*ones(25000,1);
         6*ones(25000,1);
         7*ones(25000,1);
         8*ones(25000,1);
         9*ones(25000,1);
         10*ones(25000,1);
         11*ones(25000,1);
         12*ones(25000,1);
         13*ones(25000,1);
         14*ones(25000,1);
         15*ones(25000,1);
         16*ones(25000,1);
         17*ones(25000,1);
         18*ones(25000,1);
         19*ones(25000,1);
         20*ones(25000,1)];
    pclass = [25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000 25000/500000];   %the probability of each class
    pcluster=[];                %bestpp;        %the probability of each cluster
    pclass_cluster=[];      %the probability that a member of cluster j belongs to class i
    n = length(classid);
    for i=1:20
        if i==1
            c1=1;
            c2=25000;
        elseif i==2
            c1=25001;
            c2=50000;
        elseif i==3
            c1=50001;
            c2=75000;
        elseif i==4
            c1=75001;
            c2=100000;
        elseif i==5
            c1=100001;
            c2=125000;
        elseif i==6
            c1=125001;
            c2=150000;
        elseif i==7
            c1=150001;
            c2=175000;
        elseif i==8
            c1=175001;
            c2=200000;
        elseif i==9
            c1=200001;
            c2=225000;
        elseif i==10
            c1=225001;
            c2=250000;
        elseif i==11
            c1=250001;
            c2=275000;
        elseif i==12
            c1=275001;
            c2=300000;
        elseif i==13
            c1=300001;
            c2=325000;
        elseif i==14
            c1=325001;
            c2=350000;
        elseif i==15
            c1=350001;
            c2=375000;
        elseif i==16
            c1=375001;
            c2=400000;
        elseif i==17
            c1=400001;
            c2=425000;
        elseif i==18
            c1=425001;
            c2=450000;
        elseif i==19
            c1=450001;
            c2=475000;
        elseif i==20
            c1=475001;
            c2=500000;
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