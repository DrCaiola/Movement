%Run MP4 Batch Movement Code for Movement.m with file selection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT FOLDER LOCATION
folder=uigetdir;
%INPUT FILE SAVE NAME
[name,path]=uiputfile(['batch',datestr(now,'mmddyyyy'),'.mat']);
% Grabbing video files
ext='.MP4';
ext=['.+\',ext,'$'];
k=0;
clear file
d=dir(folder);
for i=3:length(d)
    if regexpi(d(i).name,ext)
        k=k+1;
        file{k}=[folder, '\', d(i).name];
    end
end
if k==0
    warning('No MP4 found, trying WMV')
    ext='.wmv';
    for i=3:length(d)
    if regexpi(d(i).name,ext)
        k=k+1;
        file{k}=[folder, '\', d(i).name];
    end
    end
end
if k==0
    error('No files found')
end
%% Running batch
V=Movement(file,[],[],1,[1,1,1],[],'rectangle');
save([path,name],'V');