
% Updated by 
% Mohamed Gresha
% 2021
base_path = 'E:\1-Paper\PAPER 2\11\2- second_big\2';
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

% %================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
% % en1=1500;
% for i=1 :10
%     DD(i).data=X([st1:st1+7999,st2:st2+7999,st3:st3+7999,st4:st4+7999,st5:st5+7999],:)';
%     st1=st1+8000;
%     st2=st2+8000;
%     st3=st3+8000;
%     st4=st4+8000;
%     st5=st5+8000;
%     [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
%     P(i).NN=NN;
%     P(i).bestk=bestk;
%     P(i).ACL=ACL;
%     P(i).clusterid=clusterid;
%     file_name=sprintf('acl_s%d.mat',i);
%     save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P');
% end 
% %================================================================================================

%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,150001,300001];
% en1=1500;
for i=1 :150
    DD(i).data=X([st(1):st(1)+999,st(2):st(2)+999,st(3):st(3)+999],:)';
    st(1)=st(1)+1000;
    st(2)=st(2)+1000;
    st(3)=st(3)+1000;
end 

for q=1:10
    tic
    a=randi(150,1);
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(a).data');      % Competitive neural network
    P6(q).NN=NN;
    P6(q).bestk=bestk;
    P6(q).ACL=ACL;
    P6(q).clusterid=clusterid;
    P6(q).I=a;
    time1=toc;
    P6(q).Time=time1;
    file_name=sprintf('0_acl_s%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P6'); 
end
%================================================================================================

[val6,idx] = max([P6.bestk]);
for j=1:10
    tic
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val6);     %sequention processing    
     PP6(j).NN=NN;
     PP6(j).NMI=NMI;
     PP6(j).Bestk=bestk;
     PP6(j).ACL=ACL;
     PP6(j).Clusterid=clusterid;
     time2=toc;
     PP6(j).Time=time2;
    file_name=sprintf('1_acl_s%d.mat',j);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','PP6');
end
