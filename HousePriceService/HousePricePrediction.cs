using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.IO;

namespace HousePriceService
{
    public class HousePricePrediction
    {
        public static string PredictSinglePrice(HouseData houseData, string outputModelPath = @"C:\Users\junwi\Source\Repos\HousePriceService\HousePriceService\bin\Debug\netcoreapp2.2\MLNETModels\housePriceModel.zip")
        {
            MLContext mlContext = new MLContext(seed: 0);
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
            return string.Format("Predicted Price is {0}", pv);
        }
    }
}