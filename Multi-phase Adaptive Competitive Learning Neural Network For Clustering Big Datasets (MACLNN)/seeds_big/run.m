base_path = 'D:\1-Paper\PAPER 2\11\seeds_big\1';
load ('seeds_big.mat');

[X,varmin,varrange]=atscale(X);

%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,7001,14001];
en=[7001,14001,21001];
% en1=1500;
for i=1 :500
    
    if st(1)+300>en(1) || st(2)+300>en(2) || st(3)+300>en(3)
        break;
    end
    DD(i).data=X([st(1):st(1)+299,st(2):st(2)+299,st(3):st(3)+299],:)';
    
    st(1)=st(1)+300;
    st(2)=st(2)+300;
    st(3)=st(3)+300;
end
for q=1:10
    tic
    a=randi(i,1);
    [bestk,bestw,ACL,clusterid,NN]=atacl(DD(a).data');      % Competitive neural network
    D_se(q).NN=NN;
    D_se(q).bestk=bestk;
    D_se(q).ACL=ACL;
    D_se(q).clusterid=clusterid;
    D_se(q).Data_Num=a;
    time1=toc;
    D_se(q).Time=time1;
    file_name=sprintf('acl_D%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D_se');
end
%================================================================================================

[val_se,idx] = max([D_se.bestk]);
if val_se==3
    for j=1:10
        tic
        [bestk,bestw,ACL,clusterid,NMI,NN]=demo3(val_se);     %sequention processing
        DD_se(j).NN=NN;
        DD_se(j).NMI=NMI;
        DD_se(j).Bestk=bestk;
        DD_se(j).ACL=ACL;
        DD_se(j).Clusterid=clusterid;
        time2=toc;
        DD_se(j).Time=time2;
        file_name=sprintf('acl_DD%d.mat',j);
        save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DD_se');
    end  
end