using Microsoft.ML;
using Microsoft.ML.Data;

using static Microsoft.ML.DataOperationsCatalog;

string data_path = Path.Combine(Environment.CurrentDirectory, "Data", "IMDB_Dataset.txt");

MLContext mlContext = new MLContext();

TrainTestData splitDataView = LoadData(mlContext);

ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

Evaluate(mlContext, model, splitDataView.TestSet);

UseModelWithSingleItem(mlContext, model);

UseModelWithBatchItems(mlContext, model);

TrainTestData LoadData(MLContext mlContext)
{
    //Loads the data
    IDataView dataview = mlContext.Data.LoadFromTextFile<SentimentData>(data_path, hasHeader: false);

    //Splits the loaded dataset into train and test datasets.
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataview, testFraction: 0.2);

    //Returns the split train and test datasets.
    return splitDataView;
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    //Extracts and Transforms the data
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    // Trains the model
    Console.WriteLine("=============== Create and Train the Model =================");

    // Predicts sentiment based on test data
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of the Training =================");
    Console.WriteLine();

    //Returns the model
    return model;
}

void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)
{
    // Evaluate
    Console.WriteLine("================ Evaluateing Model accuracy with the Test Data ==============");
    IDataView predictions = model.Transform(splitTestSet);

    //Evaluate the model
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    // Display the metrics 
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}"); // Accuracy metrics gets the accuracy of a model, which is proportion of correct predictions in the test
    Console.WriteLine($"Auc: {metrics.AreaUnderPrecisionRecallCurve:P2}"); // AreaunderROC curve how confident the model is correctly classifying the positive and negative classes.
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}"); // F1 Score metrics which is a balance btw prediction and recall.
    Console.WriteLine("================= End of model evaluation ================");
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    //PredictionEngine is a convient API which allows you to perform a prediction on a single instance of data.
    //PredictionEngine is not a thread safe
    //For improved performance and thread safety in production enviornments, use PredictionEnginePool service, which creates an ObjectPool of PredictionEngine objects.
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    SentimentData samplestatement = new SentimentData { SentimentText = "This was a very bad steak" };

    var resultPrediction = predictionFunction.Predict(samplestatement);

    //Display the metrics
    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();

}

void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments = new[]
    {
    new SentimentData
    {SentimentText = "This was a Horrible movie" },
     new SentimentData
    {SentimentText = "I Love after interval" }
    };



    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);

    // Use model to predict whether comment data is Positive (1) or Negative (0).
    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");

}

