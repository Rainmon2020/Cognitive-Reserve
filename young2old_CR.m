clc;
clear;

maindir = 'E:\CR_result\AGING'
cd (maindir)
filepath = dir(fullfile(maindir,'time*'));
subpath = dir(fullfile(maindir,filepath(1).name,'pred','sub*'));
    
%% reconstruction of timeseries for each subjects
for hemis = {'lh', 'rh'}
    hemisphere = hemis{1}
    for i = 1:length(subpath)
        surface_interest = sprintf('sub-AGING%04d',i)
        data_recon = [];
        for t = 1:length(filepath)
            data_recon = [];
            recon = readNPY(sprintf('time_%d/pred/sub-%d_%s_pred.npy',t,i,hemisphere));
            data_recon = cat(1,data_recon,recon);
        end
        all_var = sprintf('result/%s_rfMRI_recon-%s.mat',surface_interest,hemisphere);
        save(all_var,"data_recon");
    end
end

%%
clear;
maindir = 'E:\'
cd (maindir)

T = 240;

for i = 1:length(label)
    surface_interest=label{i};
    %dis_sum=[];
    norm_sum = [];
    sum_sum = [];
    for hemis = {'lh', 'rh'}
        hemisphere = hemis{1}

        % Load cortex mask
        cortex = dlmread(sprintf('mask/fs_cortex-%s_mask.txt', hemisphere));
        cortex_ind = find(cortex);

        parc_name = 'Schaefer400';
        parc = dlmread(sprintf('parcellations/fsaverage_%s-%s.txt', parc_name, hemisphere));
        num_parcels = length(unique(parc(parc>0)));

        data = load(sprintf('rfmri_fs/%s_rfMRI_timeseries-%s.mat', surface_interest, hemisphere));
        data_to_reconstruct = data.timeseries;
        data2 = load(sprintf('result/recon_result/%s_rfMRI_recon-%s.mat', surface_interest, hemisphere));
        data_recon = data2.data_recon;
       
        norm_total = [];
        sum_total = [];
        for network = 1:200
            cortex_ind = find(ismember(parc, network));
            % Calculate empirical FC
            data_parc_emp = data_to_reconstruct(cortex_ind,:);

            FC_emp = data_parc_emp'*data_parc_emp;
            FC_emp = FC_emp/T;

            data_parc_recon = data_recon(cortex_ind,:);

            FC_recon_temp = data_parc_recon'*data_parc_recon;
            FC_recon_temp = FC_recon_temp/T;

            norm_emp = norm(FC_emp, 'fro');
            norm_recon = norm(FC_recon_temp, 'fro');
            norm_value = norm_emp-norm_recon;
            norm_total = cat(1,norm_total, norm_value);

            % 计算特征向量矩阵
            [V_emp, D1] = eig(cov(FC_emp));
            [V_recon, D2] = eig(cov(FC_recon_temp));
            % Fisher 变换
            Fisher_emp = FC_emp * V_emp;
            Fisher_recon = FC_recon_temp * V_recon;
            % 求和
            sum_emp = sum(abs(Fisher_emp), 'all');
            sum_recon = sum(abs(Fisher_recon), 'all');

            sum_value = sum_emp - sum_recon;
            sum_total = cat(1,sum_total, sum_value);
        end
        norm_sum = cat(1,norm_sum,norm_total)%得到每个被试400个网络的距离
        sum_sum = cat(1,sum_sum,sum_total)%得到每个被试400个网络的距离
        norm_normalized = (norm_sum - mean(norm_sum)) / std(norm_sum);
        sum_normalized = (sum_sum - mean(sum_sum)) / std(sum_sum);
        value_sum = cat(2,norm_normalized,sum_normalized);
    end
    all_var = sprintf('result/network/%s-CR_result.mat',surface_interest);
    save(all_var,"value_sum");
end

%%
frequency = sum(result_total,2)./size(result_total,2);
value = mean(value_total,2);

z_fre = zscore(frequency);
z_val = zscore(value);

z = z_val-z_fre;

r_total=[];
p_total=[];
for i = 1:400
    fre = result_total(i,:);
    val = value_total(i,:);
    [r,p] = corr(fre',val');
    r_total = cat(1,r_total,r);
    p_total = cat(1,p_total,p);
end
[R,P] = corr(result_total,value_total);

q_values = mafdr(p_total);
label = find(p_total == 0)
