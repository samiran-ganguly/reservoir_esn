% A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab.
% by Mantas Lukosevicius 2012
% http://minds.jacobs-university.de/mantas

% load the data
trainLen = 2000;
testLen = 2000;
initLen = 100;
ttot = initLen+trainLen+testLen;
rand( 'seed', 42 );


image_trn = [ 0 0 0 0 0;
               0 1 1 1 0;
               0 1 0 1 0;
               0 1 1 1 0;
               0 1 0 1 0;
               0 1 1 1 0;
               0 0 0 0 0;
               ];
           
image_trn_trans = reshape(image_trn,numel(image_trn),1);
image_tst = [ 0 0 0 0 0;
               0 1 1 1 0;
               0 1 0 1 0;
               0 1 0 1 0;
               0 0 0 1 0;
               0 0 0 1 1;
               0 0 0 0 0;
               ];
image_tst_trans = reshape(image_tst,numel(image_tst),1);

dtt = repmat(image_trn_trans,1,initLen+trainLen);
data_trn = dtt + (2*rand(length(image_trn_trans),initLen+trainLen));
dtts = repmat(image_tst_trans,1,testLen);
data_tst = dtts +(2*rand(length(image_tst_trans),testLen));
% generate the ESN reservoir
inSize = length(image_trn_trans); outSize = length(image_trn_trans);
resSize = 1500;
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
mse = sum(sum((dtts(:,1:errorLen)-Y(:,1:errorLen)).^2))./errorLen;
disp( ['MSE = ', num2str( mse )] );

%plots

figure;
imagesc(reshape(dtts(:,errorLen+7),7,5));title('Original');hxx = gca; hxx.Visible = 'off';
figure;
imagesc(reshape(data_tst(:,errorLen+7),7,5));title('Distorted');hxx = gca; hxx.Visible = 'off';
figure;
imagesc(reshape(Y(:,errorLen+7),7,5));title('Filtered');hxx = gca; hxx.Visible = 'off';


