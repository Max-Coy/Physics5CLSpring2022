close all
%Violet Light
vLambda = 405e-9;
stopVoltV = -1*[-1.15, -1.09, -1.09, -1.10, -1.11];

%Green Laser
gLambda = 532e-9;
stopVoltG = -1*[-0.443, -0.485, -0.464, -0.424, -0.455];

%Red Laser
rLambda = 634.6e-9;
stopVoltR = -1*[-0.052, -0.063, -0.045, -0.050, -0.068];

e = 1.609e-19;
c = 2.998e8;

Vmean = e*mean(stopVoltV);
Gmean = e*mean(stopVoltG);
Rmean = e*mean(stopVoltR);

Verr = e*sqrt((std(stopVoltV)/sqrt(5))^2+(0.003)^2);
Gerr = e*sqrt((std(stopVoltG)/sqrt(5))^2+(0.003)^2);
Rerr = e*sqrt((std(stopVoltR)/sqrt(5))^2+(0.003)^2);

Vf = c/vLambda;
Gf = c/gLambda;
Rf = c/rLambda;

y = [Vmean,Gmean,Rmean];
x = [Vf,Gf,Rf];
yerr = [Verr,Gerr,Rerr];

P = polyfit(x,y,1);

hold on
en = 8e14;
st = 4.5e14;

plot([st,en],[P(1)*st+P(2),P(1)*en+P(2)])
errorbar(x,y,yerr,'.')
title('Frequency vs. K_m_a_x')
xlabel('Frequency (m)')
ylabel('K_m_a_x (J)')

maxslope = ((Vmean+Verr)-(Rmean-Rerr))/(Vf-Rf);
minslope = ((Vmean-Verr)-(Rmean+Rerr))/(Vf-Rf);

u1 = polyfit([Rf,Vf],[Rmean-Rerr,Vmean+Verr],1);
l1 = polyfit([Rf,Vf],[Rmean+Rerr,Vmean-Verr],1);