%Run MP4/wmv Movement Code for Movement.m with SINGLE file selection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT FOLDER LOCATION
[file,folder]=uigetfile('*.MP4;*.wmv');
%INPUT FILE SAVE NAME
[name,path]=uiputfile(['move',datestr(now,'mmddyyyy'),'.mat']);
%INPUT start
t0=inputdlg('Start Time (in seconds): ');
%INPUT end
t1=inputdlg('End Time (in seconds): ');
%% Running Code
V=Movement([folder,file],[eval(t0{1}) eval(t1{1})],[],1,[1,1,1],[],'rectangle');
save([path,name],'V');