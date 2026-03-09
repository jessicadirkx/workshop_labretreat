function [Occupancy,lpnt,lpnt_noact,fstat] = dolpntPET_ROI(frame_time,frame_dur,ont,offt,TACs,CurPET,ROI)
%lp-ntPET analysis
% Original code by Jarkko Johanson based on Normandin et al. (2012)
%   adapted by Filip Grill (2022)
% Requires FreeSurfer to be sourced by MATLAB
% and function generateBasisFunctions.m
% and function gamma_variate_madsen.m
%
%   Inputs:
%   frame_time - vector of the onset of each PET frame
%   frame_dur - vector of the duration of each PET frame
%   ont - define the time in minutes to start looking for displacement
%   offt - define the time in minutes to stop looking for displacement
%   TACs - table with time activity curves extracted from FreeSurfer
%   recon_all parcellation
%   CurPET - path to preprocessed 4D PET data
%   ROI - path to binary ROI to extract TAC from
%
%   Outputs:
%   Occupancy - Occupancy curve for best fitted function
%   lpnt - lp-ntPET fit
%   lpnt_noact - lp-ntPET fit if no displacement occurs

%% Set up times and weights
Times = [frame_time;frame_time+frame_dur]'/60;
tmid=mean(Times,2);
t_points = length(tmid);
dt = [tmid(1); tmid(2:length(tmid))-tmid(1:length(tmid)-1)];
weights = ones(1,t_points); % Can be changed to weight frames

%% Read ROI
ROI = MRIread(ROI);
[ra,rb,rc] = size(ROI.vol);
ROI_v = reshape(ROI.vol,ra*rb*rc,1)';
ROI_inx = find(ROI_v == 1);


%% Generate basis functions
Options.FunctionName = 'gamma'; % use 'gamma', could also be exponential or box
Options.gamma_alpha = [.25,0.5,1,2,4]; % "sharpness" or steepness of decay
BF = generateBasisFunctions(Times, ont, offt, Options); % define basis functions

%% Solve SRTM for baseline and fix k2'
reftac = (TACs.RightCerebellumCortex+TACs.LeftCerebellumCortex)/2;
mreftac  = [reftac(1)/2; (reftac(2:end)+reftac(1:end-1))/2];
ASRTM = zeros(t_points ,3);
ASRTM(:,1)  = reftac(1:t_points);

for k = 1:t_points
    ASRTM(k,2)  = sum(mreftac(1:k).*dt(1:k));
end

roitac = (TACs.RightPutamen+TACs.LeftPutamen+TACs.RightCaudate+ ...
    TACs.LeftCaudate+TACs.RightAccumbensarea+ ...
    TACs.LeftAccumbensarea)/6; % Whole striatum
mroitac  = [roitac(1)/2; (roitac(2:end)+roitac(1:end-1))/2];

for k = 1:t_points
    ASRTM(k,3)  = -sum(mroitac(1:k).*dt(1:k));
end

[parest se mse]   = lscov(ASRTM,roitac,weights(1:t_points));
modelfit=ASRTM*parest;
k2p=parest(2)/parest(1); % Set k2 ref region aka k2'
BP=parest(2)/parest(3)-1;

%% Set up SRTM2
ASRTM2 = zeros(t_points ,2);

for k = 1:t_points
    ASRTM2(k,1) = reftac(k) +  k2p*sum(mreftac(1:k).*dt(1:k));
end

refauc = sum(mreftac.*dt); % area under the curve ref region

DynPET = MRIread(CurPET);
temp = size(DynPET.vol);
ImgData = reshape(DynPET.vol,prod(temp(1:3)),temp(4));
mImgData = [ImgData(:,1)/2, (ImgData(:,2:end)+ImgData(:,1:end-1))/2];
AUCImgData = sum(mImgData*dt,2);

roitac = mean(ImgData(ROI_inx,:),1);
roitac = roitac';
mroitac  = [roitac(1)/2; (roitac(2:end)+roitac(1:end-1))/2];
for k = 1:t_points
    ASRTM2(k,2)  = -sum(mroitac(1:k).*dt(1:k));
end

mroitac  = [roitac(1)/2; (roitac(2:end)+roitac(1:end-1))/2];

for k = 1:t_points
    ASRTM2(k,2)  = -sum(mroitac(1:k).*dt(1:k));
end
%LSQ-estimation using lscov
[parest se_srtm2 mse_srtm2]   = lscov(ASRTM2,roitac,weights);
BP_srtm2=parest(1)*k2p/parest(2)-1;
k2_srtm2=parest(1)*k2p;
se_BP_srtm2=abs(BP_srtm2)*sqrt((se_srtm2(1)/parest(1))^2+(se_srtm2(2)/parest(2))^2)*k2p;
Fit_srtm=ASRTM2*parest;

if BP_srtm2/se_BP_srtm2>1   % checks that BP is different from 0
    %% Prepare for lp-ntPET
    % Set the Activation part of A
    % Analyze basis function set
    ActFuns = [];
    AAct = [];
    Aidx = 1;
    ntasks = length(ont); % Number of activations to consider
    for taskIdx = 1:ntasks
        for columnIdx = 1:size(BF(:,:,taskIdx),2)
            if ~isempty(find(BF(:,columnIdx,taskIdx)))
                %% Set current task activation function (null for others)
                ActFuns(:,taskIdx,Aidx) = BF(:,columnIdx,taskIdx);
                ActFuns(:,setdiff(1:ntasks,taskIdx),Aidx) = 0;
                roitac_act = roitac.*BF(:,columnIdx,taskIdx);
                mroitac_act = [roitac_act(1)/2; (roitac_act(2:end)+roitac_act(1:end-1))/2];
                AAct(:,:,Aidx) = zeros(t_points,ntasks);
                for k = 1:t_points
                    AAct(k,taskIdx,Aidx) = -sum(mroitac_act(1:k).*dt(1:k));
                end
                Aidx = Aidx+1;
                
                %% Set combined activation functions
                if ntasks >= taskIdx+1
                    AAct(:,:,Aidx) = zeros(t_points,ntasks);
                    for taskIdx2 = taskIdx:n_tasks
                        ActFuns(:,taskIdx2,Aidx) = BF(:,columnIdx,taskIdx2);
                        roitac_act = roitac.*BF(:,columnIdx,taskIdx2);
                        mroitac_act = [roitac_act(1)/2; (roitac_act(2:end)+roitac_act(1:end-1))/2];
                        for k = 1:t_points
                            AAct(k,taskIdx2,Aidx) = -sum(mroitac_act(1:k).*dt(1:k));
                        end
                    end
                    Aidx = Aidx+1;
                end
            end
        end
    end
    
    parest_all = [];
    se_all = [];
    mse_all = [];
    
    for ai = 1:Aidx-1  % generate all activation function fits
        
        [parest se mse] = lscov([ASRTM2, AAct(:,:,ai)],roitac,weights);
        parest_all(:,ai) = parest;
        mse_all(ai) = mse;
        se_all(:,ai) = se;
    end
    
    [min_mse, opti] = min(mse_all); % select the best activation function
    best_parest = parest_all(:,opti);
    best_se = se_all(:,opti);
    lpnt = [ASRTM2, AAct(:,:,opti)]*best_parest;
    lpnt_noact = ASRTM2 * best_parest(1:2);
    fstat = ((mse_srtm2-min_mse)./(ntasks+2-2))./(min_mse./(t_points-(ntasks+2))); % f value relative to srtm2
    
    best_actfun = zeros(t_points,1);
    k2 = best_parest(1)*k2p;
    k2a = best_parest(2);
    par_shift = 2;
    for idx = 1:ntasks
        G = best_parest(par_shift+idx);
        best_actfun = best_actfun+G*ActFuns(:,idx,opti);
    end

    % Calculate dynamic BP
    DBP = k2./(k2a + best_actfun) - 1;
    Occupancy = ((DBP(1)-DBP)/DBP(1))*100;
    

end
end

