**1. Feature selection, model selection and Tuning - Concrete Strength Prediction**
   - **Goal:** To predict the concrete strength. Apply feature engineering and model tuning to obtain a score above 85%
      - **Project link:** [Concrete Strength Prediction](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/Project%204_Concrete_Strength_Himaja_Sol.ipynb)
        * The concrete compressive strength is the regression problem and concrete compressive strength of concrete is a highly nonlinear function of age and ingredients
        * Applied EDA and feature engineering and transforming features
        * Tried multiple ML algs (OLS, LR, Lasso, Ridge, Polynomial, KNN, SVR, DT Regressor, RF Reg, XGBoost, Gradient Boost Reg)
        * Used KFold Cross validation to evaluate model performance and Model tuning using Hyper params
        * Performance metrics used to select Best model is R2, RMSE, MAE
          - Best Model is Gradient Boosting with 90% R2, Test MAE 3.24 and Test RMSE of 4.89
          - Computed 95% confidence interval for test RMSE

        * ***[Strength Vs other Features](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/strengthtvsothers.PNG)***
        ![Strength Vs other Features](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/strengthtvsothers.PNG)

        * ***[Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/modelcomp.PNG)***
        
        ![Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/modelcomp.PNG)

        * ***[Pred vs Actual](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/predVsActual.PNG)***
        
        ![Pred vs Actual](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/predVsActual.PNG)

        * ***[Conclusion](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/conclusion.PNG)*** 
        ![con](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/FMST%20-%20Concrete%20Strength%20Prediction/conclusion.PNG)
