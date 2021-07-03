%
% Updated by 
% Mohamed Gresha
% 2021
%
base_path = 'E:\1-Paper\PAPER 2\11\4- fourth data_big\2';
load ('d1.mat');

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
st=[1,180001,337501];
en=[180000,337500,450000];
% en1=1500;
for i=1 :500
    DD(i).data=X([st(1):st(1)+899,st(2):st(2)+899,st(3):st(3)+899],:)';
    if st(1)+900>en(1) || st(2)+900>en(2) || st(3)+900>en(3)
        break;
    end
    %     st(1)=st(1)+800;
    %     st(2)=st(2)+800;
    %     st(3)=st(3)+800;
    %     st(4)=st(4)+800;
    %     st(5)=st(5)+800;
    st=st+900;
end
for q=1:10
    tic
    a=randi(i,1);
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(a).data');      % Competitive neural network
    P8(q).NN=NN;
    P8(q).bestk=bestk;
    P8(q).ACL=ACL;
    P8(q).clusterid=clusterid;
    P8(q).Data_Num=a;
    time1=toc;
    P8(q).Time=time1;
    file_name=sprintf('0_acl_s%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P8');
end
%================================================================================================
[val8,idx] = max([P8.bestk]);
for j=1:10
    tic
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val8);     %sequention processing
    PP8(j).NN=NN;
    PP8(j).NMI=NMI;
    PP8(j).Bestk=bestk;
    PP8(j).ACL=ACL;
    PP8(j).Clusterid=clusterid;
    time2=toc;
    PP8(j).Time=time2;
    file_name=sprintf('1_acl_s%d.mat',j);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','PP8');
end
