**1. Computer Vision**
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
