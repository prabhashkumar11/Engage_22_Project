# Engage_22_Project

# Face-Product-Recommendation-System

# Problem Statement: 
In this busy running world, everyone is so busy from the start of the day until we hit the bed. If we take up a case like every day we will go for supermart for bringing up things we will make so fast for checking out things and when we come to our body or face products mainly facewash which we will pick up randomly and use them which leads unknowingly dermal problems so I want to develop a software/ model which can recognize the skin type of the people and recommends the facewash based on skin type.
# Solution:
We will build a Neural Network model(CNN) architecture that classifies the skin types. we will use predefined or URL-based face products which are dedicated to specific types of skin so we can recommend these products to users after classifying the skin type.
# Techologies/ Software used : 
TensorFlow, OpenCV.
# Idea of Approach:
We want to collect the data from the internet of different skin types and we want to use OpenCV haar cascade classifier/ ROI(Region Of Interest) haar cascade classifier for detecting the faces or we will use Region Of Interest for cropping the images manually. We may use the data augmentation for increasing the training data when we have low resources for data then we will create a model which classifies the skin type and we will build a recommendation system.
# Additional Context: 
There are around four different skin types 1.Normal skin 2. Oily skin 3. Dry skin 4. Combination Skin
We need at least 100 pictures of each kind of skin type so we can make multiply them using data augmentation techniques.
## Prior information is that at least we can work on two different skin types as we go on can further we can upgrade to further skin types.

# Working Demonstration of Notebooks:
You can download all the respective files in your single working directory so that you cannot face any difficulty while working with a model.
We have “face_wash_tensorflow.ipynb” is the main notebook in which all the model architecture is built and all data pre-processing tasks have done at each level and you can easily understand every cell of the notebook I have added some theoretical explanations of the steps so that everyone can easily go through them.

I have taken two different types of skin named Oily, Normal skin and created a dataset using these two classes of images. Actually, these images are recreated from the original images as we can see there’s another notebook in this repository named “ROI.ipynb” which is actually an OpenCV methodological cropping techniques are demonstrated in this notebook in which I have commented at every step of the code so we can easily understand the workflow.

## Basically, this notebook will take an image and open up a window-sized frame which allows us to select the region of interest of the image. I have used this technique to crop the surroundings of the image so that we can have only face.


