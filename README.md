# Pneumonia-Classification

The goal is to identify Pneumonia in X-ray photos of lungs. A Convolutional Neural Network was used for the classification process, and it's accuracy on average is 90%. 

I found that the accuracy and learning capability of a CNN is primarily dependent on the number of parameters available in the model, which comes with adding more channels to later feature maps. Changing the optimizer (from SGD to ADAM) in this case, is another factor deciding the model's learning capability, as the model was able to learn with ADAM. The model seldom learned with the SGD optimizer enabled, though I used various learning rates in the range (1e-5,100). A smaller learning rate led to lack of learning (and low testing accuracy), and the learning rate of 100 enabled a low training loss (but low testing accuracy). The learning rate also affects the speed of convergence. I also noticed that while adding normalization (Batch Normalization), the training improves further. The normalization is the factor that increased the training accuracy from an 85% to 90% (on average).


NOTE: The dataset used for training had a balanced amount of Normal vs. Pneumonia samples.
Dataset: https://www.kaggle.com/datasets/assemelqirsh/chest-x-ray-dataset
