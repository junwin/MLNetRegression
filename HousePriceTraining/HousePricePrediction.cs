using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace myApp
{
    public class HousePricePrediction
    {
        /// <summary>
        /// Use a trained model to predict a house sale price
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


        public static string PredictSinglePrice(HouseData[] houseData, MLContext mlContextx, string dataTransformModelPath = @"C:\Users\junwi\source\repos\MLNetRegression\HousePriceService\bin\Debug\netcoreapp2.2\MLNETModels\housePriceDataTransformer.zip", string outputModelPath = @"C:\Users\junwi\source\repos\MLNetRegression\HousePriceService\bin\Debug\netcoreapp2.2\MLNETModels\housePriceModel.zip")
        {
            MLContext mlContext2 = new MLContext();
            
            // Create a deat view from the house data objects
            IDataView data = mlContext2.Data.LoadFromEnumerable< HouseData > (houseData);

            // Define data preparation and trained model schemas
            DataViewSchema dataPrepPipelineSchema, modelSchema;

            // Load data preparation pipeline and trained model
            ITransformer dataPrepPipeline = mlContext2.Model.Load(dataTransformModelPath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext2.Model.Load(outputModelPath, out modelSchema);


            // Transform inbound data
            var transformedData = dataPrepPipeline.Transform(data);

            // Use transform to produce an set of predictions
            var predictedPrices = trainedModel.Transform(transformedData);

            // Print out the prediced prices
            var scoreColumn = predictedPrices.GetColumn<float>("Score");
            foreach (var r in scoreColumn)
                Console.WriteLine(r);
           

            return "";
        }
    }
}