# Cognitive-Reserve

1. CR_recon_young.py
Implements the construction of a Rejuvenation-Oriented Structural-Functional Brain Prediction Model by establishing multivariate associations between cortical thickness patterns and functional connectivity profiles in young adults.

2. young2old.m
Generates age-normed functional predictions for elderly populations using the rejuvenation-oriented predictive model, then quantifies Cognitive Reserve (CR) values through Frobenius norm discrepancies between reconstructed and empirically observed functional connectivity matrices.

3. abeta_anova_predict_model.m
Performs ANOVA-based CR comparisons across clinical groups (CN/MCI/AD) and constructs LOOCV regression models to predict Aβ deposition patterns from CR spatial distributions, with feature selection based on RMSE reduction thresholds.

