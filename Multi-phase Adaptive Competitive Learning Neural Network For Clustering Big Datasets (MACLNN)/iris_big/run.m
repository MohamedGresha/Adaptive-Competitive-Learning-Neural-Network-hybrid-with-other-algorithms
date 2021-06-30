base_path = 'D:\1-Paper\PAPER 2\11\iris_big\1';
load ('atiris_big.mat');

[X,varmin,varrange]=atscale(X);


%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,5001,10001];
en=[5000,10000,15000];
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
    D_ir(q).NN=NN;
    D_ir(q).bestk=bestk;
    D_ir(q).ACL=ACL;
    D_ir(q).clusterid=clusterid;
    D_ir(q).Data_Num=a;
    time1=toc;
    D_ir(q).Time=time1;
    file_name=sprintf('0_acl_D%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D_ir');
end
%================================================================================================

[val_ir,idx] = max([D_ir.bestk]);
if val_ir==3
    for j=1:10
        tic
        [bestk,bestw,ACL,clusterid,NMI,NN]=demo3(val_ir);     %sequention processing
        DD_ir(j).NN=NN;
        DD_ir(j).NMI=NMI;
        DD_ir(j).Bestk=bestk;
        DD_ir(j).ACL=ACL;
        DD_ir(j).Clusterid=clusterid;
        time2=toc;
        DD_ir(j).Time=time2;
        file_name=sprintf('1_acl_DD%d.mat',j);
        save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DD_ir');
    end
end