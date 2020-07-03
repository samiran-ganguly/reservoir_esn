% A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab.
% by Mantas Lukosevicius 2012
% http://minds.jacobs-university.de/mantas

% load the data
trainLen = 5000;
testLen = 2000;
initLen = 100;
ttot = initLen+trainLen+testLen;

%  dtt = sin(ttot*ones(ttot,1)/600);
q = zeros(ttot,1);
 dtt = sign(sin(2*pi*(1:ttot)/600)+0.5*cos(2*pi*(1:ttot)/790)+0.7*tan(2*pi*(1:ttot)/330)+0.2*sec(2*pi*(1:ttot)/930));
for n = 8:ttot-2
    q(n) = 0.08*dtt(n+2) - 0.12*dtt(n+1) + dtt(n) + 0.18*dtt(n-1) - 0.1*dtt(n-2);
end
data = q + 0.36*q.^2 - 0.11*q.^3 + 3*(1-2*rand(ttot,1));


% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 20;
a = 0.8; % leaking rate
b = 0.003;
g = 1;

rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;
W = rand(resSize,resSize)-0.5;
W = 0.5*(W+W');
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
% W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* ( 1.2 /rhoW) + 0.5*eye(resSize);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t);
    x = (1-a)*x + a*tanh( g*( Win*[1;u] + W*x) ) + b*(1-2*rand(resSize,1));
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output
reg = 1e-8;  % regularization coefficient
X_T = X';
% Wout = Yt*X_T/(X*X_T + reg*eye(1+inSize+resSize));
 Wout = Yt*pinv(X);

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh(g*( Win*[1;u] + W*x) ) + b*(1-2*rand(resSize,1));
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
% 	u = y;
	% this would be a predictive mode:
	u = data(trainLen+t+1);
end

errorLen = 500;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );


% plot some signals
% figure;
% plot( dtt(trainLen+2:trainLen+testLen+1), 'color', [0,0.85,0] );
% hold on;
% plot( data(trainLen+2:trainLen+testLen+1), 'color', [0.85,0.05,0] );
% hold on
% plot( Y', 'b' );
% legend('Original','Distorted','Recovered');
% hold off;
% axis tight;
% xlim(1000,1500);

nImages = 800;
im = cell(nImages,1);
fig = figure('rend','painters','pos',[100 100 600 600]);
for idx = 1:nImages
    subplot(311);plot( dtt(trainLen+2:trainLen+idx+1), 'r' );title('Original');ylim([-1.5 1.5]);
    subplot(312);plot( data(trainLen+2:trainLen+idx+1), 'g' );title('Distorted');ylim([-5 5]);
    subplot(313);plot( Y(1:idx)', 'b' );title('Recovered');ylim([-1.5 1.5]);
    drawnow;
    frame = getframe(fig);
    im{idx} = frame2im(frame);
end
close;
filename = 'ESN_filtering_1D.gif'; % Specify the output file name
for idx = 1:nImages
    [A,map] = rgb2ind(im{idx},256);
    if idx == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1e-2);
    else
       imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1e-2);
    end
end
