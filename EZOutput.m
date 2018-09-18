%Export median numbers
%INPUT SAVEFILE
[savefile,folder]=uigetfile('*.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d=load([folder,'\',savefile]);
if size(d.V,2)==1
    m=[d.V.filtered.median_m d.V.filtered.median_m_sign d.V.filtered.median_m_z];
    fprintf('The median m score = %e \n',m(1))
    fprintf('The median msign score = %e \n',m(2))
    fprintf('The median mz score = %e \n',m(3))
else
    for i=1:size(d.V,2)
        m=[d.V{1,i}.filtered.median_m d.V{1,i}.filtered.median_m_sign d.V{1,i}.filtered.median_m_z];
        in=regexp(d.V{1,i}.meta_data.filename,'\');
        name=d.V{1,i}.meta_data.filename(in(end)+1:end-4);
        fprintf(['The median m score for ' name ' = %e \n'],m(1))
        fprintf(['The median msign score for ' name ' = %e \n'],m(2))
        fprintf(['The median mz score for ' name ' = %e \n'],m(3))
    end
end