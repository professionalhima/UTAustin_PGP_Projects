**1. Unsupervised Machine Learning - AllLife Credit Card Customer Segmentation**
   - Covers K-means clustering, Hierarchical clustering techniques with different linkages and PCA
      - **Project link:** [AllLife Credit Card Customer Segmentation](https://nbviewer.jupyter.org/github/sharmapratik88/AIML-Projects/blob/master/04_Unsupervised%20Learning/04_Unsupervised%20Learning.ipynb)
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
