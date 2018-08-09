function V = Movement(fname,seg_times,frame_interval,save_flag,smoothing_factors,oldparam,varargin)
% With resize
% This routine calculates differences between frames in a video stream. The
% frames are first converted into grey scale format, followed by
% calculation of the mean pixel value difference (output m), the number of
% pixels that change beyond a certain statistical level (m_sign). There is
% also a z-scored version of the m values (m_z).  The output parameter t
% (under each of the sub-fields in V) is the time axis (in seconds). Data
% are generated in the original format, as well as a median filtered format.
% The filter setting is determined as described below (should be set to the
% lowest value that removes erroneous peaks). Note that there are 2
% different filters to be set!
%   The routine generates an output structure that contains the various
% versions of the data, along with a meta_data section that contains
% descriptions of the video object, the frame boundaries chosen for the
% analysis, and other information. The data can be saved (set the
% 'save_flag' to 1). This will overwrite previously existing data from the
% same file. The input parameter seg_times describes the video segment that
% will be analyzed.  For example, [20 40] would mean that data between the
% 20th and 40th second of the video will be analyzed.  An empty vector ([])
% means that the entire data stream will be analyzed. The frame interval is
% a single number, referring to the interval between analyzed video frames
% (in seconds). If left empty ([]), the original frame rate will be used.
% The input 3-element vector smoothing_factors specifies the smoothing that
% will be applied to the filtered m, m_sign, and m_z values, respectively.
% If set to [], the filtering will be interactive.
%
% Example use:
% V = Movement('20160513185229.wmv',[10 50],0.25,1,[2 1 1]);
%
% This will analyze the data between the 10th and 50th second, and save the
% results of structure V in a file named 20160513185229_analysis.mat.
% Frames in 0.25s intervals will be analyzed. The output will be smoothed
% with a moving median filter of total width 5 (2+2+1) for the m-values,
% and 3 (1+1+1) for the m_sign and m_z values;
%
% Additional Features: 1) Use same rectangle as a previous V by supplying it
% as the 6th input. 2) Unforce a square aspect ratio by supplying
% 'rectangle' as the 7th input. 3) Allow multiple files to be run, if in
% cell format.
%
% To account for different video resolutions, results are now normalized by
% picture resolution - making these resuts incompatible with older
% versions.
%
% Another Example use:
% V=Movement('video.wmv',[],[],1,[1,1,1],[],'rectangle');
%
% This will analyze the entirety of the video, and save the
% results of structure V in a file named video_analysis.mat.
% Frames in will be analyzed in the native video framerate. The output will
% be smoothed with a moving median filter of total width 3 (1+1+1). The
% region of interest will be a rectangle.
%
% Batch Example use:
% V=Movement({'video1.wmv','video2.wmv'},[],[],1,[1,1,1],[],'rectangle');
%
% This will analyze both video1.wmv and video2.wmv, and save the
% results of structure V in files named video1_analysis.mat and video2_analysis.mat.
% Video length will be determined by the smallest of the max(n-1,1)st file
% or the length of file n.
% Frames in will be analyzed in the native video framerate. The output will
% be smoothed with a moving median filter of total width 3 (1+1+1). The
% region of interest will be a rectangle.
%
% Initial Parameters Example use:
% V=video_processing_v28M({'video1.wmv','video2.wmv','video3.wmv'});
%
% This will allow the user to select regions of interest on multiple videos
% for future use. V will be a cell housing meta data of each region that
% can be added as an input later.
%
%
% Note: This routine was developed with Matlab version 9.3.0.713579
% (R2017b).  Versions earlier than this may not work properly.
%
% Written 06/07/2018 MC
% Previous Versions Written 06/14/2016-07/08/2016, 7/30/2016-8/13/2016 TW, 10/01/2016-10/12/2016 MC, 1/13/2017 MC, 1/10/18 MC, 4/17/18 MC, 5/29/18 MC
%%

%% Housekeeping
rect=0;
opt_initial=0;
opt_bar=0;
TOL=10^-10;
nTOL=.1;
if exist('progressbar','file')==2
    opt_bar=1;
end
if nargin>=6 && ~isempty(oldparam)
    if iscell(oldparam)
        oldV=oldparam;
        oldparam=1;
    else
        oldV{1}=oldparam;
        oldparam=1;
    end
    
end
if nargin<=5 || isempty(oldparam)
    oldparam=0;
end
if nargin==1
    opt_initial=1;
    frame_interval=[];
    seg_times=[];
    rect=1;
end
numvarargs= length(varargin);
if numvarargs~=0; k=1;
    while k<=numvarargs
        switch varargin{k}
            case 'rectangle'
                rect=1;
            case 'TOL'
                TOL=varargin{k+1};
                k=k+1;
            case 'nTOL'
                nTOL=varargin{k+1};
                k=k+1;
        end
        k=k+1;
    end
end
read_all = 0;
if iscell(fname)        %allows multiple files to be run (in batch format) if cell: {'file1', 'file2',...}
    numfiles=length(fname);
    for i=1:numfiles
        vidObj(i) = VideoReader(fname{i});
        %i
    end
else
    fname={fname};
    numfiles=1;
    vidObj = VideoReader(fname{1});
end
if oldparam && ~isequal(length(oldV),numfiles)
    oldparam=0;
    warning('length(oldparam) must equal length(fname): oldparam is turned off')
end
%Resize var
Theight=480;
Twidth=720;
shr=.5;
for i=1:numfiles
    if numfiles>1 | isempty(frame_interval) | frame_interval <= 1/vidObj(i).FrameRate,
        frame_interval(i) = 1/vidObj(i).FrameRate;
        %t0=vidObj(i).CurrentTime;
        %readFrame(vidObj(i));
        %frame_interval(i) = vidObj(i).CurrentTime-t0;
        read_all = 1;
    end
    % adjust seg_times if needed
    if isempty(seg_times)
        seg_times(i,:) = [0 vidObj(i).Duration-frame_interval(i)];
    elseif size(seg_times,1)<numfiles && i>1
        seg_times(i,:) = seg_times(1,:); %seg times all set to the same time
    end
    if seg_times(i,1) < 0, seg_times(i,1) = 0; warning('video(s) starting at 0'); end
    if seg_times(i,2) > vidObj(i).Duration, seg_times(i,2) = vidObj(i).Duration-frame_interval(i); warning(['Adjusted length: i=',num2str(i)]); end
    
    %% User input
    if ~oldparam                                                                        % Allows input for same day batches
        vidObj(i).CurrentTime = seg_times(i,1);                                                 % prepare reading of first frame
        FirstFrame = readFrame(vidObj(i));                                                    % read the frame
        figure('units','normalized','outerposition',[.05 .1 .85 .9])
        image(imresize(FirstFrame,shr*[Theight,Twidth]));
        axis equal;
        if ~rect
            title({'Move and Drag The Square then push Set';[num2str(i),'/',num2str(numfiles)]})
            h=imrect(gca, [10 10 100 100]);
            setFixedAspectRatioMode(h,1)
        else
            title({'Click and Drag The Rectangle then push Set';[num2str(i),'/',num2str(numfiles)]})
            if i>1
                button = uicontrol('Style','pushbutton','units','normalized','position', [0 0 0.2 0.2],'String',...
                    'Set','Callback','uiresume(gcbf)');
                h=imrect(gca, pos);
            else
                h=imrect;
                button = uicontrol('Style','pushbutton','units','normalized','position', [0 0 0.2 0.2],'String',...
                    'Set','Callback','uiresume(gcbf)');
            end
        end
        uiwait(gcf);
        pos=getPosition(h);
        min_x(i)=pos(1);
        min_y(i) = pos(2);
        max_x(i) = pos(1)+pos(3);
        max_y(i) = pos(2)+pos(4);
        BW{i} = createMask(h);
        close(gcf);
    else
        min_x(i)=oldV{i}.meta_data.frame_boundaries(1);
        max_x(i)=oldV{i}.meta_data.frame_boundaries(2);
        min_y(i)=oldV{i}.meta_data.frame_boundaries(3);
        max_y(i)=oldV{i}.meta_data.frame_boundaries(4);
        BW{i} = oldV{i}.meta_data.mask;
    end
    x_len(i) = max_x(i)-min_x(i);
    y_len(i) = max_y(i)-min_y(i);
    if opt_initial
        V{i}.meta_data.mask = BW{i};
        V{i}.meta_data.frame_boundaries = [min_x(i) max_x(i) min_y(i) max_y(i)];
    end
end

if ~opt_initial
    if opt_bar
        progressbar('Analyzing Video','Computing Frames') % Init 2 bars
    else
        h = waitbar(0,'Processing video frames');
    end
    for i=1:numfiles
        n=1;                                                                                     % initialize loop counter for this part of the code
        fs = NaN(1000000,1);
        m = zeros(1000000,1);
        m_sign = m;
        m_z = m;
        max_ba = m;
        ba_col = m;
        ba_row = m;
        
        videoFReader = vision.VideoFileReader(fname{i});
        %Determine first frame
        while n*frame_interval(i) < seg_times(i,1)
            videoFReader();
            n=n+1;
        end
        n = 1;
        % Process first frame
        fs(1) = seg_times(i,1);
        F_comp = double(imresize(rgb2gray(videoFReader()),shr*[Theight,Twidth]));                               % this is the entire first frame, in grayscale, and resized to the desired size
        
        % scale the first frame into the 0 -> 1 space
        F_comp = (F_comp - min(F_comp(BW{i} == 1)))/max(F_comp(BW{i} == 1));
        
        % Process all subsequent frames
        while fs(n)+frame_interval(i) < seg_times(i,2)
            n = n+1;
            % read and discard those videoframes that are not going to be analyzed
            
            % read frame, convert to grayscale, resize it, and scale it to the
            % appropriate 0 -> 1 range, based on the area of each frame outside of
            % the analysis subframe
            fs(n) = seg_times(i,1)+(n-1)*frame_interval(i);
            F_test = double(imresize(rgb2gray(videoFReader()),shr*[Theight,Twidth]));
            F_test = (F_test - min(F_test(BW{i} == 1)))/max(F_test(BW{i} == 1));
            
            % *** code for blob analysis
            %            F_t = F_test(round(min_y(i)):round(max_y(i)),round(min_x(i)):round(max_x(i)));
            %            B = find_filled_circles(~imbinarize(F_t,0.35)); % the second parameter in the imbinarize function is arbitrarily set to 0.35.  It may make sense to use other values (the higher the number, the longer it takes to process ...)
            %            [max_ba(n),ind] = max(B(:));                    % max_ba is the radius of the largest blob that could be fitted
            %            [ba_row(n),ba_col(n)] = ind2sub(size(B),ind);   % bc_col and bc_row are the image column and row values that are central to the animal
            % end of code for blob analysis
            
            global_diff = abs(F_test - F_comp);
            gd_bg = global_diff(BW{i} == 0);
            m_bg = mean(gd_bg);
            s_bg = std(gd_bg);
            %             if s_bg<10^-8
            %                 [s_bg i n]
            %             end
            
            if isequal(global_diff,zeros(size(global_diff)))
                m(n-1)=m(n-2);
                m_sign(n-1)=m_sign(n-2);
                m_z(n-1)=m_z(n-2);
                warning(['Duplicate Frame at n=', num2str(n-1)])
            elseif sum(gd_bg)/numel(F_comp)<TOL
                m(n-1)=m(n-2);
                m_sign(n-1)=m_sign(n-2);
                m_z(n-1)=m_z(n-2);
                warning(['Same bg at n=', num2str(n-1)])
            else
                T=global_diff./F_comp>nTOL; %Threshold of movement to denoise
                global_diff=T.*global_diff;
                gd_t = global_diff(BW{i} == 1);
                m(n-1) = mean(gd_t);
                m_sign(n-1) = length(gd_t(gd_t > m_bg + 2*s_bg))/numel(gd_t);
                m_z(n-1) = mean(abs(gd_t-m_bg)/s_bg);
            end
            
            % clean up
            F_comp = F_test;
            if opt_bar
                progressbar([],(fs(n)-seg_times(i,1))/diff(seg_times(i,:)))
            else
                waitbar((fs(n)-seg_times(i,1))/diff(seg_times(i,:)));
            end
        end
        if~opt_bar
            close(h);
        end
        drawnow;
        
        fs = fs(1:n);
        m = m(1:n-1);
        m_sign = m_sign(1:n-1)/(x_len(i)*y_len(i));
        m_z = m_z(1:n-1);
        t = fs(2:end);
        
        % code added for blob analysis
        %        max_ba = max_ba(1:n-1);
        %        ba_col = ba_col(1:n-1);
        %        ba_row = ba_row(1:n-1);
        % end of code for blob analysis
        
        %% Generate output structure
        
        % code for blob analysis
        %        V{i}.ba.max = max_ba;
        %        V{i}.ba.col = ba_col;
        %        V{i}.ba.row = ba_row;
        % end of code for blob analysis
        
        V{i}.original.m = m;
        V{i}.original.m_sign = m_sign;
        V{i}.original.m_z = m_z;
        V{i}.original.median_m = median(V{i}.original.m);
        V{i}.original.median_m_sign = median(V{i}.original.m_sign);
        V{i}.original.median_m_z = median(V{i}.original.m_z);
        V{i}.original.t = t;
        
        if isempty(smoothing_factors),
            ft_m = interactive_threshold_setting_v2(V{i}.original.m,t,'m');
            ft_m_sign = interactive_threshold_setting_v2(V{i}.original.m_sign,t,'m_sign');
            ft_m_z = interactive_threshold_setting_v2(V{i}.original.m_z,t,'m_z');
        else
            ft_m = smoothing_factors(1);
            ft_m_sign = smoothing_factors(2);
            ft_m_z = smoothing_factors(3);
        end
        
        V{i}.filtered.m = median_smooth(V{i}.original.m,ft_m);
        V{i}.filtered.m_sign = median_smooth(V{i}.original.m_sign,ft_m_sign);
        V{i}.filtered.m_z = median_smooth(V{i}.original.m_z,ft_m_z);
        V{i}.filtered.t = t;
        V{i}.filtered.median_m = median(V{i}.filtered.m);
        V{i}.filtered.median_m_sign = median(V{i}.filtered.m_sign);
        V{i}.filtered.median_m_z = median(V{i}.filtered.m_z);
        
        V{i}.meta_data.analysis_times = fs;
        V{i}.meta_data.desired_frame_interval = frame_interval;
        V{i}.meta_data.median_actual_frame_interval = median(diff(fs));
        V{i}.meta_data.video_information = vidObj(i);
        V{i}.meta_data.number_of_processed_frames = length(fs);
        V{i}.meta_data.frame_boundaries = [min_x(i) max_x(i) min_y(i) max_y(i)];
        V{i}.meta_data.smoothing_factors.m = ft_m;
        V{i}.meta_data.smoothing_factors.m_sign = ft_m_sign;
        V{i}.meta_data.smoothing_factors.m_z = ft_m_z;
        V{i}.meta_data.seg_times = seg_times(i,:);
        V{i}.meta_data.version = '05/24/2018';
        V{i}.meta_data.mask = BW{i};
        V{i}.meta_data.filename = fname{i};
        
        if save_flag
            out_name = [fname{i}(1:length(fname{i})-4) '_paperanalysis.mat'];
            v=V;
            clear V
            V=v{i};
            save(out_name,'V');
            clear V
            V=v;
            clear v
        end
        delete(videoFReader);
        if opt_bar
            progressbar((i)/numfiles,0)
        end
    end
    if opt_bar
        progressbar(1)
    end
    if length(V)==1
        V1=V{i};
        clear V
        V=V1;
    end
end
end
%%

% ******************* SUBROUTINES ********************
%%
function ft = interactive_threshold_setting_v2(input_vals,t,title_str)

h = figure;
plot(t,input_vals);
axis tight;
title([title_str ' values']);
ft = 0;
inpt = input('Do you want to specify smoothing parameter (Y/N)? ','s');
while ~strcmp(inpt,'n') && ~strcmp(inpt,'N'),
    ft = input('Specify smoothing parameter (0 = none): ');
    plot(t,input_vals,t,median_smooth(input_vals,ft));
    axis tight;
    title([title_str ' values, smoothing parameter: ' num2str(ft)]);
    inpt = input('Do you want to specify smoothing parameter (Y/N)? ','s');
end
close(h);
end
%%
function draw_box(min_x,max_x,min_y,max_y,c_str)

line([min_x max_x],[min_y min_y],'Color',c_str);
line([min_x max_x],[max_y max_y],'Color',c_str);
line([min_x min_x],[min_y max_y],'Color',c_str);
line([max_x max_x],[min_y max_y],'Color',c_str);

end
%%
function Y = median_smooth(X,sm_width)

X = X(:);
lenX = length(X);
Y = NaN(lenX,1);
if sm_width > lenX,
    return;
end
X = [X(sm_width:-1:1);X;X(end:-1:end-sm_width+1)];

for n = 1:lenX,
    Y(n) = median(X(n:n+2*sm_width));
end
end
function pushbutton_callback(src,event)
c=1;
end

function B = find_filled_circles(A)
% This function studies, for each point in input 2D matrix A how big of a
% circle might fit onto it

B = zeros(size(A));
r_max = min(size(A,1)/2,size(A,2)/2);
for i = 1:size(A,1)
    for j = 1:size(A,2)
        for r = 1:r_max
            if i > r  && i+r <= size(A,1) && j > r && j+r <= size(A,2)
                C = draw_filled_circle_sub_v1(r);
                if isempty(find(A(i-r:i+r,j-r:j+r)-C < 0, 1))
                    B(i,j) = r;
                else
                    break;
                end
            else
                break;
            end
        end
    end
end
end

function H = draw_filled_circle_sub_v1(r)

I = ones(r);                        % produce a square matrix of r * r elements, representing a quarter of the full circle
for n = r:-1:1
    for m = r:-1:1
        if n^2 + m^2 > r^2
            I(n,m) = 0;
        else
            break;
        end
    end
end

H = [flip(I);ones(1,r);I];
H = [fliplr(H),ones(r*2+1,1),H];

% alternative (but slower algorithm:
% x = zeros(r);
% for n = 1:r
%     x(n,:)=n;
% end;
% y=x';
% I = ones(r^2,1);
% I(x(:).^2 + y(:).^2 > r^2) = 0;
% I = reshape(I,r,r);
%
% H = [flip(I);ones(1,r);I];
% H = [fliplr(H),ones(r*2+1,1),H];

% I tried also the following, but this is much slower
% x = zeros(r);
% for n = 1:r
%     x(n,:)=n;
% end;
% y=x';
% [~,rho] = cart2pol(x,y);
% I = zeros(size(x));
% I(rho <= r) = 1;
%
% H = ones(r*2+1);
% H(r+2:end,r+2:end) = I;          % fill 'lower right hand corner' of circle
% H(1:r,r+2:end) = rot90(I);       % fill 'upper right hand corner' of circle
% H(1:r,1:r) = rot90(I,2);         % fill 'upper left hand corner' of circle
% H(r+2:end,1:r) = rot90(I,3);     % fill 'upper right hand corner' of circle

end
