This tutorial shows you how to create a .NET Core console application that classifies sentiment from website comments and takes the appropriate action. The binary sentiment classifier uses C# in Visual Studio 2022.

In this tutorial, you learn how to:

Create a console application
Prepare data
Load the data
Build and train the model
Evaluate the model
Use the model to make a prediction
See the results.

Below is the result: 
=============== Create and Train the Model =================
=============== End of the Training =================

================ Evaluateing Model accuracy with the Test Data ==============

Model quality metrics evaluation
--------------------------------
Accuracy: 79.39%
Auc: 88.13%
F1Score: 79.27%
================= End of model evaluation ================

=============== Prediction Test of model with a single sample and test dataset ===============

Sentiment: This was a very bad steak | Prediction: Negative | Probability: 0.0011423342
=============== End of Predictions ===============


=============== Prediction Test of loaded model with multiple samples ===============
Sentiment: This was a Horrible movie | Prediction: Negative | Probability: 0.21653919
Sentiment: I Love after interval | Prediction: Positive | Probability: 0.95274705
