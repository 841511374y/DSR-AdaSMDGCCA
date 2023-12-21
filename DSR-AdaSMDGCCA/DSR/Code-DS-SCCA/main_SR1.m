clear all
clc
% 未经过多层神经网络
rng(1)
%% importing data and making network parameters in an optimum state and partition Data for the 5 fold test.
X1 = xlsread('D:\\yangrenbo\\386样本\\ROI.xlsx');
X2 = xlsread('D:\\yangrenbo\\386样本\\SNP.xlsx');
X3 = xlsread('D:\\yangrenbo\\386样本\\GENE.xlsx');
label = csvread('D:\\yangrenbo\\386样本\\label.csv');
% X1= X1./repmat(sqrt(sum(X1.^2,2)),1,size(X1,2));
% X2= X2./repmat(sqrt(sum(X2.^2,2)),1,size(X2,2));
% X3= X3./repmat(sqrt(sum(X3.^2,2)),1,size(X3,2));

X1 = normalize(X1, 'minmax');
X2 = normalize(X2, 'minmax');
X3 = normalize(X3, 'minmax');

%% Use the seed to reproduce the errors listed below.
%randseed=8409;

rcov1=1e-4; rcov2=1e-4; rcov3=1e-4;
% Hidden activation type.
hiddentype='sigmoid';
% Architecture (hidden layer sizes) for genotypes data neural network.

NN1=[10240 5120 2480 1240 140]; 
NN2=[10240 5120 2957];
NN3=[10240 5120 4026];
     
% Weight decay parameter.
l2penalty=1e-4;

%% Run DCCA with SGD. No pretraining is used.
% Minibatchsize.
batchsize=5;
% Learning rate.
eta0=0.001;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=0.19
% Momentum.
momentum=0.99;
% How many passes of the data you run SGD with.
maxepoch=1000;
addpath ./deepnet/

[N,D1]=size(X1); [~,D2]=size(X2); [~,D3]=size(X3);
  %% Set genotypes data architecture.
  Layersizes1=[D1 NN1];  Layertypes1={};
  for nn1=1:length(NN1)-1
    Layertypes1=[Layertypes1, {hiddentype}];
  end
  % I choose to set the last layer to be linear.
  Layertypes1{end+1}='linear';
  %% Set brain network phenotypes data architecture.
 Layersizes2=[D2 NN2];  Layertypes2={};
  for nn2=1:length(NN2)-1
   Layertypes2=[Layertypes2, {hiddentype}];
 end
Layertypes2{end+1}='linear';

 %% Set brain network phenotypes data architecture.
 Layersizes3=[D3 NN3];  Layertypes3={};
  for nn3=1:length(NN3)-1
   Layertypes3=[Layertypes3, {hiddentype}];
 end
Layertypes3{end+1}='linear';
%% Random initialization of weights.
  F1=deepnetinit(Layersizes1,Layertypes1);
  F2=deepnetinit(Layersizes2,Layertypes2);
  F3=deepnetinit(Layersizes3,Layertypes3);

 
  for j=1:length(F1)  F1{j}.l=l2penalty;  end
  for j=1:length(F2)  F2{j}.l=l2penalty;  end
  for j=1:length(F3)  F3{j}.l=l2penalty;  end
  %% the outputs at the top layer.
  FX1=deepnetfwd(X1,F1); FX2=deepnetfwd(X2,F2); FX3=deepnetfwd(X3,F3); 
%the self-representation matrix is learned for reconstructing the source data at the top layer.
  
    options.lambda = 1;
            opts = [];
            opts.init = 0;
            opts.tFlag = 10; opts.maxIter = 100;
            opts.rFlag = 10^-5;
            opts.rsL2 = 0;
            options.opts = opts;
            options.label = label;
% 

          wSNPdata= f_SR(X1', options);
          wSNPdata=wSNPdata+wSNPdata';
     
        
          wBNdata = f_SR(X2', options);
            wBNdata=wBNdata+wBNdata';
            
            wgenedata = f_SR(X3', options);
            wgenedata=wgenedata+wgenedata';
            
            sSNPdata= wSNPdata*X1;
            sBNdata= wBNdata*X2;
            sgenedata = wgenedata*X3;
            
kk=1; 
kfold = 5;
[tcv,fcv]=f_myCV(label',kfold,kk);
for cc = 1:kfold
    trLab=tcv{cc}';
    teLab=fcv{cc}';
    X{1,cc}=sSNPdata(trLab,:);  
    Y{1,cc}=sBNdata(trLab,:);
    Z{1,cc}=sgenedata(trLab,:);
    Label{1,cc}=label(trLab,:);
    
    Xt{1,cc}=sSNPdata(teLab,:);
    Yt{1,cc}=sBNdata(teLab,:);
    Zt{1,cc}=sgenedata(teLab,:);
    Labelt{1,cc}=label(teLab,:);
    
end
xlswrite('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\ROI_re.xlsx',sSNPdata);
xlswrite('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\SNP_re.xlsx',sBNdata);
xlswrite('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\Gene_re.xlsx',sgenedata);




