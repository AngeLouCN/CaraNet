% Compute averaged dice values according to areas
% input: 1) area ratio of objects in images: [r,c] = size(img);
%                                            AreaR = sum(img(:))/(r*c);
%        2) Dice value segmented objects
%        3) Width of bins for taking average

% output: averaged area ratios, averaged dice values, 
%         standard deviation of dice

%  Note: the arrays AreaR and Dice should have the **same length**. 
%  For i-th case, AreaR(i) is the area ratio (see code comments for its computation) of objects
%  in the image of i-th segmented ground truth (the binary label), 
%  Dice(i) is the Dice coefficient computed by using the i-th predicted segmentation result
%  and its ground truth. bin_width depends on the distribution of AreaR.

function [mean_areaR, mean_dice, std_dice] = local_average(AreaR, Dice, bin_width)

if nargin<3
  bin_width = 0.05; % default bin width
end

begin = min(AreaR);

N = length(AreaR);
cell_num = fix((max(AreaR)-begin)/bin_width)+1;

groups_dice = cell(cell_num,1);
groups_area = cell(cell_num,1);

for i=1:N
    bin = fix((AreaR(i)-begin)/bin_width)+1; % bin from 1
    
    groups_area{bin}=[groups_area{bin}, AreaR(i)];
    groups_dice{bin}=[groups_dice{bin}, Dice(i)];
end

mean_areaR = zeros(cell_num,1);
mean_dice = zeros(cell_num,1);
std_dice = zeros(cell_num,1);


for j=1:cell_num
    mean_areaR(j) = mean(groups_area{j});
    mean_dice(j) = mean(groups_dice{j});
    std_dice(j) = std(groups_dice{j});
    
end

% remove NaNs
mean_areaR(isnan(mean_areaR))=[];
mean_dice(isnan(mean_dice))=[];
std_dice(isnan(std_dice))=[];

% plot
errorbar(mean_areaR,mean_dice,std_dice)
end
