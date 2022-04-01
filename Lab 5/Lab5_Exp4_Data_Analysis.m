data = [4.95,9.65;4.9,9.5;8.2,15.0;7.9,15.5;11.1,21.5;11.0,21.0];
lens = [25,25;40,40;55,55];
rangs = zeros(3,2);
uncs = zeros(3,2);
stdand = zeros(3,2);
for i = 1:3
    for j = 1:2
        a = std(data([i*2-1,i*2],j))/sqrt(2)+0.05;
        b = mean(data([i*2-1,i*2],j));
        if j == 1
            rangs(i,j) = b-a;
            uncs(i,j) = a;
            stdand(i,j) = b;
        else
            rangs(i,j) = b+a;
            uncs(i,j) = a;
            stdand(i,j) = b;
        end
    end
end

angle = atan(rangs./lens);
wl = sin(angle)/5000*1e7;

angleun = atan((stdand)./lens);
wlun = sin(angleun)/5000*1e7;

l1 = mean(wlun(:,1))
l2 =mean(wlun(:,2))

unc1 = mean(wl(:,1))-l1
unc2 = mean(wl(:,2))-l2

%%
dat = [5.4,5.6,6.05,6.3,6.7,7.1,7.6,7.9
5.5,5.65,6.2,6.65,7.0,7.5,8.0,8.4
8.6,8.8,9.50,10.00,10.65,11.20,12.05,12.40
8.75,8.9,10.80,10.40,11.05,11.6,12.85,13.1
11.4,11.65,12.8,13.45,14.2,14.7,16.1,16.45
11.6,11.9,13.1,13.9,14.6,15.3,16.95,17.4];

lens1 = [25*ones(1,8);40*ones(1,8);55*ones(1,8)];
rangs1 = zeros(3,8);
uncs1 = zeros(3,8);
stdand1 = zeros(3,8);
for i = 1:3
    for k = 1:4
    for j = 1:2
        a = std(dat([i*2-1,i*2],(k*2-2)+j))/sqrt(2)+0.05;
        b = mean(dat([i*2-1,i*2],(k*2-2)+j));
        if j == 1
            rangs1(i,(k*2-2)+j) = b-a;
            uncs1(i,(k*2-2)+j) = a;
            stdand1(i,(k*2-2)+j) = b;
        else
            rangs1(i,(k*2-2)+j) = b+a;
            uncs1(i,(k*2-2)+j) = a;
            stdand1(i,(k*2-2)+j) = b;
        end
    end
    end
end

angle1 = atan(rangs1./lens1);
wl1 = sin(angle1)/5000*1e7;

angleun1 = atan((stdand1)./lens1);
wlun1 = sin(angleun1)/5000*1e7;

wtp = mean(wlun1)

delt = mean(abs(wl1-wlun1))
