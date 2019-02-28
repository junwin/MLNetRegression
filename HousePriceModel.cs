using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace myApp
{
    public class HousePriceModel
    {
        private static string NumFeatures = nameof(NumFeatures);

        private static string CatFeatures = nameof(CatFeatures);

        /// <summary>
        /// Train and save model for predicting next month country unit sales
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <param name="outputModelPath">Trained model path</param>
        public static void TrainAndSaveModel(MLContext mlContext, string dataPath, string outputModelPath = "housePriceModel.zip")
        {
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            CreateHousePriceModelUsingPipeline(mlContext, dataPath, outputModelPath);
        }


        /// <summary>
        /// Build model for predicting next month country unit sales using Learning Pipelines API
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <returns></returns>
        private static void CreateHousePriceModelUsingPipeline(MLContext mlContext, string dataPath, string outputModelPath)
        {
            Console.WriteLine("Training product forecasting");

            // Read the sample data into a view that we can use for training
            var trainingDataView = mlContext.Data.ReadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            // create the trainer we will use  - ML.NET supports different training methods
            var trainer = mlContext.Regression.Trainers.FastTree(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);

            // Create the training pipeline, this determines how the input data will be transformed
            // We can also select the features we want to use here, the names used correspond to the porperty names in 
            // HouseData
            string[] numericFeatureNames = { "Area","Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "Floors","LotSize"};

            // We distinguish between features that are strings e.g. {"attached","detached","none") garage types and 
            // Numeric faeature, since learning systems only work with numeric values we need to convert the strings.
            // You can see that in the training pipeline we have applied OneHotEncoding to do this.
            string[] categoryFeatureNames = { "GarageType" };
            
            var trainingPipeline = mlContext.Transforms.Concatenate(NumFeatures, numericFeatureNames)
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(CatFeatures, inputColumnName: categoryFeatureNames[0]))
                .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features, NumFeatures, CatFeatures))
                .Append(mlContext.Transforms.CopyColumns(DefaultColumnNames.Label, inputColumnName: nameof(HouseData.SoldPrice)))
                .Append(trainer);

            // Split the data 90:10 into train and test sets, train and evaluate.
            var (trainData, testData) = mlContext.Regression.TrainTestSplit(trainingDataView, testFraction: 0.2);

            // Train the model.
            var model = trainingPipeline.Fit(trainData);
            // Compute quality metrics on the test set.
            var metrics = mlContext.Regression.Evaluate(model.Transform(testData));
            Helpers.PrintRegressionMetrics(trainer.ToString(), metrics);

            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numFolds: 6, labelColumn: DefaultColumnNames.Label);
            Helpers.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            // Train the model
            model = trainingPipeline.Fit(trainingDataView);

            // Save the model for later comsumption from end-user apps
            using (var file = File.OpenWrite(outputModelPath))
                model.SaveTo(mlContext, file);
        }
    }
}
