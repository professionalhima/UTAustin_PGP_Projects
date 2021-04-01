**1. Supervised Machine Learning**
   - **Context:** This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with a minimal budget.
   - **Goal:** The classification goal is to predict the likelihood of a liability customer buying personal loans.

      - **Project link:** [Thera Bank Personal Loan Campaign](https://nbviewer.jupyter.org/github/sharmapratik88/AIML-Projects/blob/master/02_Supervised%20Machine%20Learning/02_Supervised%20Machine%20Learning.ipynb)
        - Identified potential loan customers for Thera Bank using classification techniques. Compared multiple models (Logistic regression, KNN, Naive Bayes). Out of 3, KNN is the best in giving overall performance and greater Recall score
        - As the Class 1 data in the dataset is very low, applied SMOTE on minority class and it gave better Recall compared to just taking 10% of minority class
        - **Important Metric:** Here more focus towards should be towards recall because our target variable is 'Personal Loan' , i.e whether the customer is accepting the personal loan or not. And the bank wants more people to accept personal loan i.e. less number of False Negative, so that bank doesn't lose real customers who want to take loan. Hence the focus should be on increasing Recall. 
        
            * ***[IncomeDistribution](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/IncomeDistribution.PNG)***
            * ***[Credit Card Usage Vs Income](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/CCAvgVsIncome.PNG)***
            * ***[KNN ROC-AUC](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/KNN_ROC_SMOTE.png)***
            * ***[KNN Classficiation Report](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/KNN_ClassifReport.PNG)***
            
        -  **Learning Outcomes:**
            - Exploratory Data Analysis
            - Preparing the data to train models
            - Training and making predictions using classification models (LR, KNN, NB)
            - Model evaluation (COnfusion Matrix, ROC-AUC, Classification report, Classification Metrics)
            - Class Imbalance Handling (using SMOTE)
