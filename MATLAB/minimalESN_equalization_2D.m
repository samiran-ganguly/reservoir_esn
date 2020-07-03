%%%%%%% ESN based Reservoir Computing %%%%%%%%%

clc;clear all;%close all;
% load the data
trainLen = 3000;
testLen = 3000;
initLen = 300;
ttot = initLen+trainLen+testLen;
rand( 'seed', 42 );


image_trn_trans = [repmat(GetImage(3).',1,5) repmat(GetImage(13).',1,5) repmat(GetImage(9).',1,5) repmat(GetImage(7).',1,5) repmat(GetImage(12).',1,5) repmat(GetImage(5).',1,5)];

image_tst_trans = [repmat(GetImage(9).',1,5) repmat(GetImage(3).',1,5) repmat(GetImage(12).',1,5) repmat(GetImage(7).',1,5) repmat(GetImage(13).',1,5) repmat(GetImage(12).',1,5)];

dtt = repmat(image_trn_trans,1,(floor(initLen+trainLen)/30));
data_trn = dtt + (2*rand(size(dtt)));
dtts = repmat(image_tst_trans,1,floor(testLen/30));
data_tst = dtts +(2*rand(size(dtts)));
% generate the ESN reservoir
inSize = length(GetImage(1)); outSize = length(GetImage(1));
resSize = 500;
a = 0.3; % leaking rate


Win = (rand(resSize,1+inSize)-0.5) .* 1;
W = rand(resSize,resSize)-0.5;
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
W = W .* 0.1;
% Option 2 - normalizing and setting spectral radius (correct, slower):
% disp 'Computing spectral radius...';
% opt.disp = 0;
% rhoW = abs(eigs(W,1,'LM',opt));
% disp 'done.'
% W = W .* ( 1.25 /rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = dtt(:,initLen+2:trainLen+1);

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data_trn(:,t);
    x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output
reg = 1e-6;  % regularization coefficient
X_T = X';
 %Wout = Yt*X_T/(X*X_T + reg*eye(1+inSize+resSize));
Wout = Yt*pinv(X);

% run the trained ESN in a generative mode. no need to initialize here, 
X = zeros(1+inSize+resSize,testLen);
x = rand(resSize,1);
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
for t = 1:testLen 

    u = data_tst(:,t);
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
end

 errorLen = 500;
% mse = sum(sum((dtts(:,1:errorLen)-Y(:,1:errorLen)).^2))./errorLen;
% disp( ['MSE = ', num2str( mse )] );

%plots

% figure;
% k = 88;
% subplot(131);imagesc(reshape(dtts(:,errorLen+k).',5,7));title('Original');hxx = gca; hxx.Visible = 'off';
% subplot(132);imagesc(reshape(data_tst(:,errorLen+k).',5,7));title('Distorted');hxx = gca; hxx.Visible = 'off';
% subplot(133);imagesc(reshape(Y(:,errorLen+k).',5,7));title('Filtered');hxx = gca; hxx.Visible = 'off';

%animated GIF
nImages = 900;
im = cell(nImages,1);
fig = figure;
for idx = 1:nImages
    subplot(131);imagesc(reshape(dtts(:,errorLen+idx).',5,7));title('Original');hxx = gca; hxx.Visible = 'off';title('Original');
    subplot(132);imagesc(reshape(data_tst(:,errorLen+idx).',5,7));title('Distorted');hxx = gca; hxx.Visible = 'off';title('Distorted');
    subplot(133);imagesc(reshape(Y(:,errorLen+idx).',5,7));title('Filtered');hxx = gca; hxx.Visible = 'off';title('Recovered');
    drawnow;
    frame = getframe(fig);
    im{idx} = frame2im(frame);
end
% close;
% filename = 'AllTest_ESN_Eql.gif'; % Specify the output file name
% for idx = 1:nImages
%     [A,map] = rgb2ind(im{idx},256);
%     if idx == 1
%         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
%     else
%         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1e-2);
%     end
% end


