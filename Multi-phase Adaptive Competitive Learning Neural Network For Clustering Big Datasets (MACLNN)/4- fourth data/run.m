
% Updated by 
% Mohamed Gresha
% 2021
% clear;clc;
% runp_s
% clear;clc;
base_path = 'D:\1-Paper\PAPER 2\11\4- fourth data\3';
load ('d1_1.mat');

[X,varmin,varrange]=atscale(X);

%================================================================================================
st=[1,4501,10501];
en=[4500,10500,15000];
% en1=1500;
for i=1 :10
    tic
    DD(i).data=X([st(1):st(1)+499,st(2):st(2)+499,st(3):st(3)+499],:)';
    
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
    D4(i).NN=NN;
    D4(i).bestk=bestk;
    D4(i).ACL=ACL;
    D4(i).clusterid=clusterid;
    D4(i).Time=toc;
    file_name=sprintf('1_acl_D%d.mat',i);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D4');
    if st(1)+500>en(1) || st(2)+500>en(2) || st(3)+500>en(3)
        break;
    end
    st(1)=st(1)+500;
    st(2)=st(2)+500;
    st(3)=st(3)+500;
end  
%================================================================================================
[val4,idx] = max([D4.bestk]);
for j=1:10
    tic
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val4);     %sequention processing    
     DD4(j).NN=NN;
     DD4(j).NMI=NMI;
     DD4(j).bestk=bestk;
     DD4(j).ACL=ACL;
     DD4(j).clusterid=clusterid;
     DD4(j).Time=toc;
     file_name=sprintf('1_acl_DD%d.mat',j);
     save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DD4');
end
