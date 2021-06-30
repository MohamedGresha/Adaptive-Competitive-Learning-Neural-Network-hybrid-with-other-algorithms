clear;clc;
runp_s
clear;clc;
base_path = 'D:\1-Paper\PAPER 2\11\3- third data\3';
load ('d1_1.mat');

[X,varmin,varrange]=atscale(X);

% % % st=1;
% % % en=1500;
% % % for i=1 :10
% % %     DD(i).data=X([st:en],:)';
% % %     st=st+1500;
% % %     en=en+1500;
% % %     [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
% % %     P(i).NN=NN;
% % %     P(i).bestk=bestk;
% % %     P(i).ACL=ACL;
% % %     P(i).clusterid=clusterid;
% % %     file_name=sprintf('acl_s%d.mat',i);
% % %     save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P');
% % % end 
%================================================================================================
% %     i=1;
% %     DD(i).data=X(:,[5,6,9,8,4])';
% %     [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
% %         P(i).NN=NN;
% %         P(i).bestk=bestk;
% %         P(i).ACL=ACL;
% %         P(i).clusterid=clusterid;
% %         file_name=sprintf('acl_s%d.mat',i);
% %         save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P');
% %     i=2;
% %     DD(i).data=X(:,[3,10,1,7,2])';
% %     [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
% %         P(i).NN=NN;
% %         P(i).bestk=bestk;
% %         P(i).ACL=ACL;
% %         P(i).clusterid=clusterid;
% %         file_name=sprintf('acl_s%d.mat',i);
% %         save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P');

%================================================================================================
% st1=1;st2=3001;st3=7501;st4=9001;st5=12001;
% en1=3000;en2=7500;en3=9000;en4=12000;en5=15000;
st=[1,3001,7501,9001,12001];
en=[3000,75000,9000,12000,15000];
% en1=1500;
for i=1 :10
    tic
    DD3(i).data=X([st(1):st(1)+299,st(2):st(2)+299,st(3):st(3)+299,st(4):st(4)+299,st(5):st(5)+299],:)';
    
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD3(i).data');      % Competitive neural network
    D3(i).NN=NN;
    D3(i).bestk=bestk;
    D3(i).ACL=ACL;
    D3(i).clusterid=clusterid;
    D3(i).Time=toc;
    file_name=sprintf('acl_D%d.mat',i);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D3');
    if st(1)+300>en(1) || st(2)+300>en(2) || st(3)+300>en(3) || st(4)+300>en(4) || st(5)+300>en(5)
        break;
    end
    st(1)=st(1)+300;
    st(2)=st(2)+300;
    st(3)=st(3)+300;
    st(4)=st(4)+300;
    st(5)=st(5)+300;
end  
%================================================================================================
[val3,idx] = max([D3.bestk]);
for j=1:10
    tic
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val3);     %sequention processing    
     DD3(j).NN=NN;
     DD3(j).NMI=NMI;
     DD3(j).bestk=bestk;
     DD3(j).ACL=ACL;
     DD3(j).clusterid=clusterid;
     DD3(j).Time=toc;
     file_name=sprintf('0_acl_DD%d.mat',j);
     save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DD3');
end