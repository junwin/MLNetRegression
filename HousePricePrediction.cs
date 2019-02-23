using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Transforms.Normalizers;

namespace myApp
{
    public class HousePricePrediction
    {
        public static void PredictSinglePrice(HouseData houseData, MLContext mlContext, string dataPath, string outputModelPath = "housePriceModel.zip")
        {
            //  Load the prediction model we saved earlier
            ITransformer loadedModel;
            using (var stream = new FileStream(outputModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }



            var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);

            

            var prediction = predictionFunction.Predict(houseData);

            var pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted SellPrice: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");

        }

    }
}
