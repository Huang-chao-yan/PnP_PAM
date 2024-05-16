function [xo,fo] = Opt_Simu(f,x0,u,l,kmax,q,TolFun)
% ģ���˻��㷨���� f(x)����Сֵ�㣬 �� l <= x <= u
% fΪ��������x0Ϊ��ֵ�㣬l��u�ֱ�Ϊ��������������ޣ�kmaxΪ����������
% qΪ�˻����ӣ�TolFunΪ�����������
%%%%�㷨��һ�������������������ĳЩ����Ϊȱʡֵ
if nargin < 7
    TolFun = 1e-8;
end
if nargin < 6
    q = 1;
end
if nargin < 5
    kmax = 100;
end
%%%%�㷨�ڶ��������һЩ��������
N = length(x0); %�Ա���ά��
x = x0;
fx = feval(f,x); %�����ڳ�ʼ��x0���ĺ���ֵ
xo = x;
fo = fx;
%%%%%�㷨�����������е������㣬�ҳ�����ȫ����С��
for k =0:kmax
    Ti = (k/kmax)^q;
    mu = 10^(Ti*100); % ����mu
    dx = Mu_Inv(2*rand(size(x))-1,mu).*(u - l);%����dx
    x1 = x + dx; %��һ�����Ƶ�
    x1 = (x1 < l).*l +(l <= x1).*(x1 <= u).*x1 +(u < x1).*u; %��x1�޶�������[l,u]��
    disp(x1);
    fx1 = feval(f,x1);
    disp([k,fx1])
    df = fx1- fx;
    if df < 0||rand < exp(-Ti*df/(abs(fx) + eps)/TolFun) %���fx1<fx���߸��ʴ��������z
        x = x1;
        fx = fx1;
    end
    if fx < fo
        xo = x;
        fo = fx1;
    end
end
function x = Mu_Inv(y,mu)
x = (((1+mu).^abs(y)- 1)/mu).*sign(y);