%Read batch and indvidual move files
%INPUT SAVEFILE
[savefile,folder]=uigetfile('*.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~strcmpi(savefile(end-3:end),'.mat')
    savefile=[savefile,'.mat'];
end
d=load([folder,'\',savefile]);

buffer=10;
B=[.3*ones(1,3);.4*ones(1,3)];
warning('off', 'MATLAB:audiovideo:VideoReader:FileNotFound')
pt=1;
figure;
for type={'m','m_sign','m_z'}
    x_off=buffer;
    y=[];
    yt=[];
    z=[];
    zt=[];
    subplot(3,1,pt);
    hold on
    for i=1:length(d.V)
        plot(d.V{1,i}.filtered.t+x_off,d.V{1,i}.filtered.(type{1}),'Color',B(1+mod(i,2),:));
        if strcmp(type,'m_z') && isnan(d.V{1,i}.filtered.median_m_z)
            erI=find(isnan(d.V{1,i}.filtered.m_z));
            for k=erI'
                if k==1
                    d.V{1,i}.filtered.m_z(1)=0;
                end
                d.V{1,i}.filtered.m_z(k)=d.V{1,i}.filtered.m_z(k-1);
            end
            d.V{1,i}.filtered.median_m_z=median(d.V{1,i}.filtered.m_z);
        end
        y=[y d.V{1,i}.filtered.(['median_',type{1}])];
        yt=[yt median(d.V{1,i}.filtered.t+x_off)];
        z=[z; d.V{1,i}.filtered.(type{1})];
        zt=[zt; d.V{1,i}.filtered.t+x_off];
        x_off=x_off+d.V{1,i}.filtered.t(end);
        x_end=x_off;
    end
    plot(yt,y,'ro-')
    if pt==1
        m=y;
    elseif pt==2
        msign=y;
    elseif pt==3
        mz=y;
    end
    xticks(yt)
    if pt==2
        title('m_{sign}')
    else
        title(type{1})
    end
    xlim([0,x_end])
    ylim([median(z)/3, mean(z)*2.5])
    pt=pt+1;
    xticklabels({})
    xtickangle(45)
end