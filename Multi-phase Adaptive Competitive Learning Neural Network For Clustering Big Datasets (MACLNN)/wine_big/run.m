%
% Updated by 
% Mohamed Gresha
% 2021
%
base_path = 'D:\1-Paper\PAPER 2\11\wine_big\1';
load ('atwine_big.mat');

[X,varmin,varrange]=atscale(X);

%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,5901,13001];
en=[5900,13000,17800];
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
    D_wn(q).NN=NN;
    D_wn(q).bestk=bestk;
    D_wn(q).ACL=ACL;
    D_wn(q).clusterid=clusterid;
    D_wn(q).Data_Num=a;
    time1=toc;
    D_wn(q).Time=time1;
    file_name=sprintf('0_acl_D%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','D_wn');
end
%================================================================================================

[val_wn,idx] = max([D_wn.bestk]);
if val_wn==3
    for j=1:10
        tic
        [bestk,bestw,ACL,clusterid,NMI,NN]=demo3(val_wn);     %sequention processing
        DD_wn(j).NN=NN;
        DD_wn(j).NMI=NMI;
        DD_wn(j).Bestk=bestk;
        DD_wn(j).ACL=ACL;
        DD_wn(j).Clusterid=clusterid;
        time2=toc;
        DD_wn(j).Time=time2;
        file_name=sprintf('0_acl_DD%d.mat',j);
        save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid','NN','DD_wn');
    end
    
end
