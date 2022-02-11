# UTAustin_PGP_Projects

## Projects:
**1. Foundation for AIML - MovieLens Data Exploration**
   - Covers Descriptive Statistics, Exploratory Data Analysis covering Visualizations too
      - **Project link:** [MovieLens Data Exploration](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/MovieLens%20Data%20Exploration/MovieLensProject1_Solutions_HimajaM.ipynb)
         - The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. The data is widely used for collaborative filtering and other filtering solutions. However, we will be using this data to act as a means to demonstrate our skill in using Python to “play” with data.
         - **Learning Outcomes:**
         	- Exploratory Data Analysis
         	- Visualization using Python
         	- Pandas – groupby, merging

**2. Supervised Machine Learning - Thera Bank Personal loan campaign**
   - **Context:** This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with a minimal budget.
   - **Goal:** The classification goal is to predict the likelihood of a liability customer buying personal loans.

      - **Project link:** [Thera Bank Personal Loan Campaign](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/Project_2_Thera_Bank_Personal_Loan_Campaign_Himaja_Sol.ipynb)
        - Identified potential loan customers for Thera Bank using classification techniques. Compared multiple models (Logistic regression, KNN, Naive Bayes). Out of 3, KNN is the best in giving overall performance and greater Recall score
        - As the Class 1 data in the dataset is very low, applied SMOTE on minority class and it gave better Recall compared to just taking 10% of minority class
        - **Important Metric:** Here more focus towards should be towards recall because our target variable is 'Personal Loan' , i.e whether the customer is accepting the personal loan or not. And the bank wants more people to accept personal loan i.e. less number of False Negative, so that bank doesn't lose real customers who want to take loan. Hence the focus should be on increasing Recall. 
        
            * ***[IncomeDistribution](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/IncomeDistribution.PNG)***
            * ***[Credit Card Usage Vs Income](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/CCAvgVsIncome.PNG)***
            * ***[KNN ROC-AUC of 95%](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/KNN_ROC_SMOTE.png)***
            * ***[KNN Classficiation Report with 92% Recall](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/KNN_ClassifReport.PNG)***
            * ***[Models Comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/Model_comp.PNG)***
            
             ![Model Comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/Model_comp.PNG)
             ![KNN ROC](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Supervised%20Learning%20-%20Personal%20Loan%20Campaign%20Modelling/KNN_ROC_SMOTE.png)
                
        -  **Learning Outcomes:**
            - Exploratory Data Analysis
            - Preparing the data to train models
            - Training and making predictions using classification models (LR, KNN, NB)
            - Model evaluation (Confusion Matrix, ROC-AUC, Classification report, Classification Metrics)
            - Class Imbalance Handling (using SMOTE)

**3. Ensemble Techniques - Term Deposit Subscription**
   - **Goal:** To build a model that will help the marketingteam identify potential customers who are relatively more likely to subscribe to termdeposits and thus increase their hit ratio.
      - **Project link:** [Term Deposit Subscription](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/Project3_EnsembleTech_TermDepositSale_HimajaM_Sol.ipynb)
        * Covers LR, Decision Trees, Bagging, Random Forests, Boosting Algorithms. 
        * Because of class imbalance, applied SMOTE
        * Classification report and confusion matrix analysis
        * Feature Selection / Elemination - Used Recursive Feature Elemination and checked model performances for various models
          - Best Model is Gradient Boosting with 89% f1, 89% precision and 90% recall

        * ***[Best Model Details](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/Random%20Forest%20Classifier.png)***
        
        ![Scores](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/Random%20Forest%20Classifier.png)

        * ***[Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/modelcomp.PNG)***
        ![Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/modelcomp.PNG)

        * ***[Models Analysis](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/ma.PNG)***
        ![ModelsA](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/ma.PNG)

        * ***[Business Insights](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/BI.PNG)*** 
        ![BI](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/BI.PNG)

        * ***[After RFE -> Best Model](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/rfemodel.PNG)***
        ![RFE](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Ensemble%20Techniques%20-%20Term%20Deposit%20Subscription%20Prediction/rfemodel.PNG)

**4. Feature selection, model selection and Tuning - Concrete Strength Prediction**
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

**5. Unsupervised Machine Learning - AllLife Credit Card Customer Segmentation**
   - Covers K-means clustering, Hierarchical clustering techniques with different linkages and PCA
      - **Project link:** [AllLife Credit Card Customer Segmentation](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/Project_5_CC_seg_Unsupervised_HimajaM_Sol.ipynb)
         - **Objective:** To identify different segments in the existing customer based on their spending patterns as well as past interaction with the bank
         - **Key Questions to be answered?**
            - How many different segments of customers are there?
            - How are these segments different from each other?
            - What are your recommendations to the bank on how to better market to and service these customers?
         - **Approach:**
           - Identified different customer segments by applying KMeans, different Hierarchical clustering techniques
           - Choosing value of K (no. of clusters) using elbow, silhouette diagram, silhouette scores
           - Hierarchical clustering using SciPy Linkage and calculating cophenetic correlation to see how better are clusters
           - Interesting observations are made during EDA and it is proved after doing clustering
           - Used PCA in order to reduce dimensionality

         - ***[KMeans Clusters 3D view](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/clusters_KMeans.PNG)***
         
        ![KMeans Clusters 3D view](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/clusters_KMeans.PNG)
         - ***[Agglomerative Clusters 3D View](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/Agglomerative_Clusets.PNG)***
         - ***[PCA & K Means Clusters 3D view](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/PCA_KMeans.PNG)***
         - ***[Elbow method for optimal K](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/elbow.PNG)***
        ![To find optimal clusters](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/elbow.PNG)
         - ***[Sihouette scores for various K](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/SillhouetteScores_multiple.PNG)***
        ![Sihouette scores for various K](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/SillhouetteScores_multiple.PNG)
         - ***[Hierarchical Clustering view](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/agg.PNG)***
         - ***[Silhouette Diagram for optimal K](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/SillhouetteDiagram.PNG)***
        ![Silhouette Diagram for optimal K](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/SillhouetteDiagram.PNG)
         - ***[Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/models.PNG)***
        ![Models comparision](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/models.PNG)

      - **Recommendations to Bank:** 
      ![Recommendations](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/AllLife%20Credit%20Card%20Customer%20Segmentation/recommendations.PNG)         

**5. Computer Vision**
   - Covers Introduction to Convolutional Neural Networks, Convolution, Pooling, Padding & its mechanisms, Forward propagation & Backpropagation for CNNs
      - **Project link:** [Plant Seedlings Image Classification using CNNs in Keras](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/Project_7_PlantSeedlingCNN_Himaja_Sol_v1.ipynb)
         - Recognize, identify and classify plant images using CNN and image recognition algorithms. The goal of the projectis to create a classifier capable of determining a plant's species from a photo.

      - **Learning Outcomes:** 
        - Pre-processing of image data.
        - Visualization of images
        - Building CNN and Evaluate the Model
        - The motive of the project is to make the learners capable to handle images/image classification problems, during this process you shouldalso be capable to handle real image files, not just limited to a numpy array of image pixels
      - **Details:**
        - Applied Image preprocessing techniques (Resize, Gaussian Blurr, Masking, grey scale and Laplacian Edge detection)
        - CNN with Batch Normalization, Maxpooling, dropouts + Dense layers is a good combination for image classification
        - CNN Model Architecture
            - Convolutional input layer, 32 feature maps with a size of 3X3 and a * rectifier activation function
            - Batch Normalization
            - Max Pool layer with size 2×2 and a stride of 2
            - Convolutional layer, 64 feature maps with a size of 3X3 and a rectifier activation function.
            - Batch Normalization
            - Max Pool layer with size 2×2 and a stride of 2
            - Convolutional layer, 64 feature maps with a size of 3X3 and a rectifier activation function.
            - Batch Normalization
            - Max Pool layer with size 2×2 and a stride of 2
            - Flatten layer
            - Fully connected or Dense layers (with 512 and 128 neurons) with Relu Act.
            - Dropout layer to reduce overfitting or for regularization
            - O/p layer with Softwax fun. to detect multiple categories
         
      - ***[Different Plant types](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/OriginalPlantImgs.PNG)***
        ![Plant_image](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/OriginalPlantImgs.PNG)
         
      -  ***[Resize, Gaussian Blurr and Masking](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_color.PNG)*** 
         ![Preprocessing](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_color.PNG)
         
      -  ***[Gray scale](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_gray.PNG)***
         ![gray](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_gray.PNG)

      -  ***[Laplacian edge](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_egde.PNG)***
         ![edge](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/processed_egde.PNG)

      -  ***[Classification report](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/classifreport.PNG)***
      
      ![report](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/classifreport.PNG)

      -  ***[Confusion Matrix](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/confusionMtrx.png)***
       
       ![CM](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/confusionMtrx.png)

      -  ***[Sample Prediction](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/samplepred.PNG)***
      
       ![sample](https://github.com/professionalhima/UTAustin_PGP_Projects/blob/main/Plant%20Seedlings%20Image%20Classification%20using%20CNNs%20in%20Keras/samplepred.PNG)
