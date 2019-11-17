using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace HousePriceService
{
    public class HousePricePrediction
    {
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
            float[] results = new float[houseData.Length];
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

            // Print out the prediced prices
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