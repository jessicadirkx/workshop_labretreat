function [BPimg_nii,BPseimg_nii,modelfit,ASRTM_mse] = doSRTM_voxelwise(frame_time,frame_dur,TACs,CurPET)
    % SRTM2 analysis
    % Original code by Jarkko Johanson based on Normandin et al. (2012)
    %   adapted by Filip Grill (2024)
    % Requires FreeSurfer to be sourced by MATLAB
    %
    %   Inputs:
    %   frame_time - vector of the onset of each PET frame
    %   frame_dur - vector of the duration of each PET frame
    %   TACs - table with time activity curves extracted
    %   CurPET - path to preprocessed 4D PET data
    %
    %   Outputs:
    %   BPimg_nii - Binding potential nifti object
    %   BPseimg_nii - Standard error of BP calculation
    %   modelfit - SRTM fit to the striatum
    
    %% Set up times and weights
    Times = [frame_time;frame_time+frame_dur]'/60;
    tmid=mean(Times,2);
    t_points = length(tmid);
    dt = [tmid(1); tmid(2:length(tmid))-tmid(1:length(tmid)-1)];
    weights = ones(1,t_points); % Can be changed to weight frames
    
    
    %% Solve SRTM for baseline and fix k2'
    %reftac = TACs.Cerebellum_LR_C;
    reftac = (TACs.Right_Cerebellum_Cortex+TACs.Left_Cerebellum_Cortex)/2;
    mreftac  = [reftac(1)/2; (reftac(2:end)+reftac(1:end-1))/2];
    ASRTM = zeros(t_points ,3);
    ASRTM(:,1)  = reftac(1:t_points);
    
    for k = 1:t_points
        ASRTM(k,2)  = sum(mreftac(1:k).*dt(1:k));
    end
    
    %roitac = (TACs.Putamen_r_C+TACs.Putamen_l_C+TACs.CaudateNucl_r_C+ ... 
    %          TACs.CaudateNucl_l_C+TACs.Ventral_striatum_r_C+ ... 
    %          TACs.Ventral_striatum_l_C)/6; % Whole striatum
    roitac = (TACs.Right_Putamen+TACs.Left_Putamen+TACs.Right_Caudate+ ...
              TACs.Left_Caudate+TACs.Right_Accumbens_area+ ...
              TACs.Left_Accumbens_area)/6; % Whole striatum
    mroitac  = [roitac(1)/2; (roitac(2:end)+roitac(1:end-1))/2];
    
    for k = 1:t_points
        ASRTM(k,3)  = -sum(mroitac(1:k).*dt(1:k));
    end
    
    [parest se mse]   = lscov(ASRTM,roitac,weights(1:t_points)); %SRTM
    ASRTM_mse = mse;
    % parest - parameter estimation of R1, k2, and k2a
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
    maskidx = find(AUCImgData>1.75*refauc); % Determine voxels with enough activity counts for analysis
    %maskidx = find(AUCImgData>1.05*refauc); % very lenient

    Mask = zeros(size(AUCImgData),'int8');
    Mask(maskidx) = 1;
    DynPET.img = reshape(Mask,temp(1:3));
    DynPET.hdr.dime.dim(5) = 1;
    
    fprintf(1,'Starting to process %1.2f percent of voxels. \n',100*length(maskidx)/prod(temp(1:3)));
    
    %% Create outputs
    BPimg = zeros(temp(1:3));  % 3 D output file
    BPseimg = zeros(temp(1:3));
    
    fail_count = 0;
    
    for voxidx=maskidx'
        roitac=squeeze(ImgData(voxidx,:))';
        mroitac  = [roitac(1)/2; (roitac(2:end)+roitac(1:end-1))/2];
        for k = 1:t_points
            ASRTM2(k,2)  = -sum(mroitac(1:k).*dt(1:k));
        end
        %LSQ-estimation using lscov
        [parest se_srtm2 mse_srtm2]   = lscov(ASRTM2,roitac,weights); %SRTM2
        % parest - parameter estimation of R1 and k2
        BP_srtm2=parest(1)*k2p/parest(2)-1;
        k2_srtm2=parest(1)*k2p;
        se_BP_srtm2=abs(BP_srtm2)*sqrt((se_srtm2(1)/parest(1))^2+(se_srtm2(2)/parest(2))^2)*k2p;
        Fit_srtm=ASRTM2*parest;
        BPimg(voxidx) = BP_srtm2;
        BPseimg(voxidx) = se_BP_srtm2;
    
    %else
    %    fail_count=fail_count+1;
    end
    %fprintf(1,'done, with %i (%1.2f) fails\n',fail_count,100*fail_count/length(maskidx));
    BPimg_nii = DynPET;
    BPimg_nii.vol = BPimg;
    BPseimg_nii = DynPET;
    BPseimg_nii.vol = BPseimg;
end

