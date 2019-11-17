using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace myApp
{
    public class HousePricePrediction
    {
        /// <summary>
        /// Use a trained model to predict a house sale price - this works when a single pipeline
        /// has been used when creating the model
        /// </summary>
        /// <param name="houseData"></param>
        /// <param name="mlContext"></param>
        /// <param name="dataPath"></param>
        /// <param name="outputModelPath"></param>
        public static void PredictSinglePrice(HouseData houseData, MLContext mlContext, string outputModelPath = "housePriceModel.zip")
        {
            //  Load the prediction model we saved earlier
            ITransformer loadedModel;
            DataViewSchema dataViewSchema;
            using (var stream = new FileStream(outputModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream, out dataViewSchema);
            }

            // Create a handy function based on our HouseData class and a class to contain the result
            //var predictionFunction = loadedModel.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(loadedModel, dataViewSchema);

            // Predict the Sale price - TA DA
            var prediction = predictionFunction.Predict(houseData);

            var pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted SellPrice: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");
        }

        /// <summary>
        /// This takes a set of features and predicts a set of prices. It assumes taht a separate pipeline was used for
        /// data preparation and another for traing the model - for example using cross validation
        /// </summary>
        /// <param name="houseData"></param>
        /// <param name="mlContextx"></param>
        /// <param name="dataTransformModelPath"></param>
        /// <param name="outputModelPath"></param>
        /// <returns></returns>
        public static float[] PredictSinglePriceSet(HouseData[] houseData, string dataTransformModelPath, string outputModelPath)
        {
           
            // create a new context
            MLContext mlContext = new MLContext();

            // Create a data view from the house data objects
            IDataView data = mlContext.Data.LoadFromEnumerable<HouseData>(houseData);

            // Define data preparation and trained model schemas
            DataViewSchema dataPrepPipelineSchema, modelSchema;

            // Load data preparation pipeline and trained model
            ITransformer dataPrepPipeline = mlContext.Model.Load(dataTransformModelPath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(outputModelPath, out modelSchema);

            // Transform inbound data
            var transformedData = dataPrepPipeline.Transform(data);

            // Use transform to produce an set of predictions
            var predictedPrices = trainedModel.Transform(transformedData);

            // get the result set and print out the values
            float[] results = new float[houseData.Length];
            var scoreColumn = predictedPrices.GetColumn<float>("Score");

            int i = 0;
            foreach (var r in scoreColumn)
            {
                Console.WriteLine(r);
                results[i++] = r;
            }

            return results;
        }
    }
}