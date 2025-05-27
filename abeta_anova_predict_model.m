maindir = 'E:\CR_result\abeta';
cd(maindir);
load('result.mat');

z_ad_fre = zscore(ad_fre_group);
z_ad_val = zscore(ad_val_group);
z_mci_fre = zscore(mci_fre_group);
z_mci_val = zscore(mci_val_group);
z_cn_fre = zscore(cn_fre_group);
z_cn_val = zscore(cn_val_group);

r_cn=[];
p_cn=[];
for i = 1:400
    fre = cn_value(i,:);
    val = cn_fre(i,:);
    [r,p] = corr(fre',val');
    r_cn = cat(1,r_cn,r);
    p_cn = cat(1,p_cn,p);
end
r_mci=[];
p_mci=[];
for i = 1:400
    fre = mci_value(i,:);
    val = mci_fre(i,:);
    [r,p] = corr(fre',val');
    r_mci = cat(1,r_mci,r);
    p_mci = cat(1,p_mci,p);
end
r_ad=[];
p_ad=[];
for i = 1:400
    fre = ad_value(i,:);
    val = ad_fre(i,:);
    [r,p] = corr(fre',val');
    r_ad = cat(1,r_ad,r);
    p_ad = cat(1,p_ad,p);
end
q_cn = mafdr(p_cn);
q_mci = mafdr(p_mci);
q_ad = mafdr(p_ad);

label_cn = find(q_cn<0.001);
label_mci = find(q_mci<0.001);
label_ad = find(q_ad<0.001);

%%
data = readtable('total.xlsx');
age = data.age;
gender = data.gender;
edu = data.edu;
% 中心化协变量
age_cen = bsxfun(@minus, age, mean(age));
gender_cen = bsxfun(@minus, gender, mean(gender));
edu_cen = bsxfun(@minus, edu, mean(edu));
X = [age_cen, gender_cen, edu_cen];  % 合并自变量
% 构建回归模型
residuals_total = [];
for i = 1:400
    data = table(age_cen, gender_cen, edu_cen, value_total(i,:)', ...
        'VariableNames', {'Age', 'Gender', 'Edu', 'Value'});
    % 进行回归（自动处理分类变量）
    mdl = fitlm(data, 'Value ~ Age + Gender + Edu');
    % 提取残差
    residuals = mdl.Residuals.Raw;
    residuals_total = cat(2,residuals_total,residuals);
end
residuals_total = residuals_total';

group = data.cognitive_change;
cn = find(group==0);
mci = find(group==1);
ad = find(group==2);

cn_value = value_total(:,cn);
mci_value = value_total(:,mci);
ad_value = value_total(:,ad);

cn_resi = residuals_total(:,cn);
mci_resi = residuals_total(:,mci);
ad_resi = residuals_total(:,ad);

cn_resi = residuals_total(CR_label,cn);
mci_resi = residuals_total(CR_label,mci);
ad_resi = residuals_total(CR_label,ad);

%% anova
p_cn_mci = [];
f_cn_mci = [];
c_cn_mci = [];
for i  = 1:400
    % 示例数据（替换为你的实际数据）
    Group1 = cn_resi(i,:);
    Group2 = mci_resi(i,:);
    Group3 = ad_resi(i,:);

    % 合并数据并创建分组标签
    data = [Group1, Group2, Group3];
    groups = [repmat({'Group1'}, 1, length(Group1)), ...
        repmat({'Group2'}, 1, length(Group2)), ...
        repmat({'Group3'}, 1, length(Group3))];

    % 单因素方差分析
    [p, tbl, stats] = anova1(data, groups, 'off');
    [c, m, ~, gnames] = multcompare(stats, 'CType', 'tukey-kramer', 'Display', 'off');
    c2 = [c,[i;i;i]];
    p_cn_mci = cat(1,p_cn_mci,p);
    f_cn_mci = cat(1,f_cn_mci,tbl{2,5});
    c_cn_mci = cat(3,c_cn_mci,c2);
end
label_cn_mci_ad = find(p_cn_mci<0.05);
c_cn_mci_ad = c_cn_mci(:,:,label_cn_mci_ad);
B = permute(c_cn_mci_ad, [1, 3, 2]); % 调整维度顺序为3×32×6
c_shaped = reshape(B, [96, 7]); % 展开为96×6
label_cn_mci_ad = find(c_shaped(:,6)<0.05);
c_cn_mci_ad = c_shaped(label_cn_mci_ad,:);
label_cn_mci = find(c_cn_mci_ad(:,2)==2);
c_cn_mci = c_cn_mci_ad(label_cn_mci,:);
label_cn_ad = find(c_cn_mci_ad(:,2)==3);
c_cn_ad = c_cn_mci_ad(label_cn_ad,:);

parcel_cn_mci = label
parcel_cn_ad =
% 使用multcompare函数（基于Tukey方法）
figure;
%[c, m, ~, gnames] = multcompare(stats, 'CType', 'tukey-kramer', 'Display', 'off');
[c, m, ~, gnames] = multcompare(stats, 'CType', 'tukey-kramer', 'Display', 'off');
disp('Tukey HSD结果:');
disp([gnames(c(:,1)), gnames(c(:,2)), num2cell(c(:,3:6))]);

%% 分组CN MCI AD预测
data = readtable("total.xlsx");
group = data.cognitive_change;
cn = find(group==0);
mci = find(group==1);
ad = find(group==2);

cn_resi = residuals_total(:,cn);
mci_resi = residuals_total(:,mci);
ad_resi = residuals_total(:,ad);

cn_abeta = abeta(:,cn);
mci_abeta = abeta(:,mci);
ad_abeta = abeta(:,ad);

pred_cn = [];
error_cn = [];
% 留一法线性预测
for n = 1:400
    X = cn_resi(n,:);
    Y = cn_abeta(n,:);

    % 初始化存储预测结果的向量
    predictions = zeros(size(Y));

    for i = 1:length(Y)
        % 留一法：第i个样本作为测试集，其余作为训练集
        trainIdx = [1:i-1, i+1:length(Y)];
        testIdx = i;

        % 训练线性回归模型
        mdl = fitlm(X(trainIdx), Y(trainIdx));

        % 预测被留出的样本
        predictions(testIdx) = predict(mdl, X(testIdx));
    end

    % 计算性能指标
    mse = mean((Y - predictions).^2);
    rmse = sqrt(mse);
    r2 = 1 - sum((Y - predictions).^2)/sum((Y - mean(Y)).^2);
    error = [mse,rmse,r2];
    error_cn = cat(1,error_cn,error);
    pred_cn = cat(2,pred_cn,predictions);
end
baseline_rmse = [];
for i = 1:400
    % 计算均值模型的RMSE
    Y = cn_abeta(i,:)'; % 确保为列向量
    baseline_pred = mean(Y)*ones(size(Y));
    baseline = sqrt(mean((Y - baseline_pred).^2));
    baseline_rmse = cat(1,baseline_rmse,baseline);
end

rmse_diff = error_cn(:,2)-baseline_rmse;
label_pred_cn = find(rmse_diff<0);
rmse_pred_cn = 1./error_cn(label_pred_cn,2);

pred_mci = [];
error_mci = [];
% 留一法线性预测
for n = 1:400
    X = mci_resi(n,:);
    Y = mci_abeta(n,:);

    % 初始化存储预测结果的向量
    predictions = zeros(size(Y));

    for i = 1:length(Y)
        % 留一法：第i个样本作为测试集，其余作为训练集
        trainIdx = [1:i-1, i+1:length(Y)];
        testIdx = i;

        % 训练线性回归模型
        mdl = fitlm(X(trainIdx), Y(trainIdx));

        % 预测被留出的样本
        predictions(testIdx) = predict(mdl, X(testIdx));
    end

    % 计算性能指标
    mse = mean((Y - predictions).^2);
    rmse = sqrt(mse);
    r2 = 1 - sum((Y - predictions).^2)/sum((Y - mean(Y)).^2);
    error = [mse,rmse,r2];
    error_mci = cat(1,error_mci,error);
    pred_mci = cat(2,pred_mci,predictions);
end
baseline_rmse_mci = [];
for i = 1:400
    % 计算均值模型的RMSE
    Y = mci_abeta(i,:)'; % 确保为列向量
    baseline_pred = mean(Y)*ones(size(Y));
    baseline = sqrt(mean((Y - baseline_pred).^2));
    baseline_rmse_mci = cat(1,baseline_rmse_mci,baseline);
end

rmse_diff = error_mci(:,2)-baseline_rmse_mci;
label_pred_mci = find(rmse_diff<0);
rmse_pred_mci = 1./error_cn(label_pred_mci,2);

pred_ad = [];
error_ad = [];
% 留一法线性预测
for n = 1:400
    X = ad_resi(n,:);
    Y = ad_abeta(n,:);

    % 初始化存储预测结果的向量
    predictions = zeros(size(Y));

    for i = 1:length(Y)
        % 留一法：第i个样本作为测试集，其余作为训练集
        trainIdx = [1:i-1, i+1:length(Y)];
        testIdx = i;

        % 训练线性回归模型
        mdl = fitlm(X(trainIdx), Y(trainIdx));

        % 预测被留出的样本
        predictions(testIdx) = predict(mdl, X(testIdx));
    end

    % 计算性能指标
    mse = mean((Y - predictions).^2);
    rmse = sqrt(mse);
    r2 = 1 - sum((Y - predictions).^2)/sum((Y - mean(Y)).^2);
    error = [mse,rmse,r2];
    error_ad = cat(1,error_ad,error);
    pred_ad = cat(1,pred_ad,predictions);
end
baseline_rmse_ad = [];
for i = 1:400
    % 计算均值模型的RMSE
    Y = mci_abeta(i,:)'; % 确保为列向量
    baseline_pred = mean(Y)*ones(size(Y));
    baseline = sqrt(mean((Y - baseline_pred).^2));
    baseline_rmse_ad = cat(1,baseline_rmse_ad,baseline);
end

rmse_diff = error_ad(:,2)-baseline_rmse_ad;
label_pred_ad = find(rmse_diff<0);
rmse_pred_ad = 1./error_cn(label_pred_ad,2);
