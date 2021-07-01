base_path = 'E:\1-Paper\PAPER 2\11\1-first data_big\4';
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
st=[1,80001,160001,240001,320001];
% en1=1500;
for i=1 :100
    DD(i).data=X([st(1):st(1)+799,st(2):st(2)+799,st(3):st(3)+799,st(4):st(4)+799,st(5):st(5)+799],:)';
    st(1)=st(1)+800;
    st(2)=st(2)+800;
    st(3)=st(3)+800;
    st(4)=st(4)+800;
    st(5)=st(5)+800;
end
for q=1:10
    tic
    i=randi(100,1);
    [bestk,bestw,ACL,clusterid,NN]=atacl_d(DD(i).data');      % Competitive neural network
    P5(q).NN=NN;
    P5(q).bestk=bestk;
    P5(q).ACL=ACL;
    P5(q).clusterid=clusterid;
    P5(q).Data_Num=i;
    time1=toc;
    P5(q).Time=time1;
    file_name=sprintf('0_acl_s%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','P5');
end
%================================================================================================

[val5,idx] = max([P5.bestk]);
if val5==5
    for j=1:10
        tic
        [bestk,bestw,ACL,clusterid,NMI,NN]=demo_s(val5);     %sequention processing
        PP5(j).NN=NN;
        PP5(j).NMI=NMI;
        PP5(j).Bestk=bestk;
        PP5(j).ACL=ACL;
        PP5(j).Clusterid=clusterid;
        time2=toc;
        PP5(j).Time=time2;
        file_name=sprintf('2_acl_s%d.mat',j);
        save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','PP5');
    end
    
end
