base_path = 'D:\1-Paper\PAPER 2\11\1-first data_test\0\GPU';
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
% st1=1;st2=3001;st3=6001;st4=9001;st5=12001;
st=[1,3001,6001,9001,12001];
% en1=1500;
for i=1 :10
    tic
    DD(i).data=X([st(1):st(1)+299,st(2):st(2)+299,st(3):st(3)+299,st(4):st(4)+299,st(5):st(5)+299],:)';
    st(1)=st(1)+300;
    st(2)=st(2)+300;
    st(3)=st(3)+300;
    st(4)=st(4)+300;
    st(5)=st(5)+300;
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
    D(i).NN=NN;
    D(i).bestk=bestk;
    D(i).ACL=ACL;
    D(i).clusterid=clusterid;
    D(i).Time=toc;
    file_name=sprintf('2_acl_D%d.mat',i);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D');
end 
%================================================================================================
[val,idx] = max([D.bestk]);
for j=1:1
    tic
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val);     %sequention processing    
     DDD(j).NN=NN;
     DDD(j).NMI=NMI;
     DDD(j).bestk=bestk;
     DDD(j).ACL=ACL;
     DDD(j).clusterid=clusterid;
     DDD(j).Time=toc;
     file_name=sprintf('2_acl_DD%d.mat',j);
     save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DDD');
end