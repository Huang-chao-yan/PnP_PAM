function[Absorption,Emission]=MakeTarget(n)

% Constructing an object for X-ray tomography
N = n^2;
t = linspace(0,1,n);
[X,Y] = meshgrid(t',t);
XY = [X(:)';Y(:)'];

% Object 1: an ellipse defining the outer boundary
c = [1/2;1/2];
theta = pi/2;
ss = sin(theta);
cc = cos(theta);
R = [cc , -ss; ss, cc];
D = diag([60,120]);
aux1 = exp(-0.5*sum((sqrt(D)*R*((XY - c*ones(1,N)))).^2));
aux1 = 0.3*(aux1>0.005);

% Object 2: inner boundary of the skull
aux2 = exp(-0.5*sum((sqrt(D)*R*((XY - c*ones(1,N)))).^2));
aux2 = 0.3*(aux2>0.01);

% Object 3: ventricles
c = [6/10;5/8];
theta = -pi/6;
ss = sin(theta);
cc = cos(theta);
R = [cc , -ss; ss, cc];
D = diag([100,20]);
aux3 = exp(-0.5*sum((sqrt(D)*R*((XY - c*ones(1,N)))).^2));
aux3 = 0.8*(aux3>0.9);

c = [4/10;5/8];
theta = pi/8;
ss = sin(theta);
cc = cos(theta);
R = [cc , -ss; ss, cc];
D = diag([100,20]);
aux4 = exp(-0.5*sum((sqrt(D)*R*((XY - c*ones(1,N)))).^2));
aux4 = 0.8*(aux4>0.87);

Absorption = reshape(aux1-0.5*aux2 -0.1*aux3 -0.1*aux4,n,n);

% Emission coefficient
c = [42/100;52/100];
theta = pi/8;
ss = sin(theta);
cc = cos(theta);
R = [cc , -ss; ss, cc];
D = diag([100,80]);
aux5 = exp(-0.5*sum((sqrt(D)*R*((XY - c*ones(1,N)))).^2));
aux5 = 0.8*(aux5>0.85);

Emission = reshape(0.4*((aux5 - aux4)>0),n,n);
