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
        public static void PredictSinglePrice(HouseData houseData, MLContext mlContext, string dataPath, string outputModelPath = "housePriceModel.zip")
        {
            //  Load the prediction model we saved earlier
            ITransformer loadedModel;
            using (var stream = new FileStream(outputModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Creete a handy function based on our HouseData class and a class to contain the result
            var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);

            // Predict the Sale price - TA DA
            var prediction = predictionFunction.Predict(houseData);

            var pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted SellPrice: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}