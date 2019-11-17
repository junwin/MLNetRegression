using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.IO;
using System.Linq;

namespace myApp
{
    public class HousePriceModel
    {
        private static string NumFeatures = nameof(NumFeatures);

        private static string CatFeatures = nameof(CatFeatures);

        
        /// <summary>
        /// Build and train the model used to predict house prices
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataPath"></param>
        /// <param name="outputModelPath"></param>
        public static void CreateHousePriceModelUsingPipeline(MLContext mlContext, string dataPath, string outputModelPath)
        {
            Console.WriteLine("Training house sell price model");



            // Load sample data into a view that we can use for training - ML.NET provides support for
            // many different data types.
            var trainingDataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            // create the trainer we will use  - ML.NET supports different training methods
            // some trainers support automatic feature normalization and setting regularization
            // ML.NET lets you choose a number of different training alogorithms
            var trainer = mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features");

            // Feature Selection - We can also select the features we want to use here, the names used
            // correspond to the porperty names in HouseData
            string[] numericFeatureNames = { "Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "ApproxSquFeet", "GarageSpaces", "ParkingSpaces" };

            // We distinguish between features that are strings e.g. {"attached","detached","none") garage types and
            // Numeric faeature, since learning systems only work with numeric values we need to convert the strings.
            // You can see that in the training pipeline we have applied OneHotEncoding to do this.
            // We have added area, since although in the data set its a number, it could be some other code.
            string[] categoryFeatureNames = { "GarageType", "Area" };

            // ML.NET combines transforms for data preparation and model training into a single pipeline, these are then applied
            // to training data and the input data used to make predictions in your model.
            /*
            var trainingPipeline = mlContext.Transforms.Concatenate(NumFeatures, numericFeatureNames)
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding("CatGarageType", inputColumnName: categoryFeatureNames[0]))
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding("CatArea", inputColumnName: categoryFeatureNames[1]))
                 .Append(mlContext.Transforms.Concatenate("Features", NumFeatures, "CatGarageType", "CatArea"))
                 .Append(mlContext.Transforms.CopyColumns("Label", inputColumnName: nameof(HouseData.SoldPrice)))
                 //.Append(mlContext.Transforms.NormalizeMinMax("Features"));
                .Append(trainer);
                */

            var trainingPipeline = mlContext.Transforms.Categorical.OneHotEncoding("CatGarageType", inputColumnName: categoryFeatureNames[0])
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding("CatArea", inputColumnName: categoryFeatureNames[1]))
                 .Append(mlContext.Transforms.Concatenate("Features", "Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "ApproxSquFeet", "GarageSpaces", "ParkingSpaces", "CatGarageType", "CatArea"))
                .Append(trainer);

            // Train the model
            var model = trainingPipeline.Fit(trainingDataView);

            // Save the model for later comsumption from end-user apps
            using (var file = File.OpenWrite(outputModelPath))
                mlContext.Model.Save(model, trainingDataView.Schema, file);
        }

        public static void CreateHousePriceModelUsingCrossValidationPipeline(MLContext mlContext, string dataPath, string dataTransformModelPath, string outputModelPath)
        {
            Console.WriteLine("Training house sell price model");

            // Load sample data into a view that we can use for training - ML.NET provides support for
            // many different data types.
            var trainingDataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            // create the trainer we will use  - ML.NET supports different training methods
            // some trainers support automatic feature normalization and setting regularization
            // ML.NET lets you choose a number of different training alogorithm

            var trainer = mlContext.Regression.Trainers.Sdca();
            
            // Feature Selection - We can also select the features we want to use here, the names used
            // correspond to the porperty names in HouseData
            // We distinguish between features that are strings e.g. {"attached","detached","none") 
            // and normal numeric features, since learning systems we use here only work 
            // with numeric values we need to convert the strings when setting up the pipeline
            // You can see that in the training pipeline we have applied OneHotEncoding to do this.
            // We have added area, since although in the data set its a number, it could be some other code
            // The pipeline below works with transforms the data to the for we can use in a
            // separate pipline for cross validation - often you can combine these into a single pipeline see above
            var dataTransformPipeline = mlContext.Transforms.Categorical.OneHotEncoding("CatGarageType", inputColumnName: "GarageType")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CatArea", inputColumnName: "Area"))
                .Append(mlContext.Transforms.Concatenate("Features", "Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath"
                , "ApproxSquFeet", "GarageSpaces", "ParkingSpaces", "CatGarageType", "CatArea"));
        
            //  We use cross-valdiation to estimate the variance of the model quality from one run to another,
            // it and also eliminates the need to extract a separate test set for evaluation.
            // We display the quality metrics in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");

            // data transform pipeline to convert he HouseData taining feature set
            var transformer = dataTransformPipeline.Fit(trainingDataView);
            var transformedData = transformer.Transform(trainingDataView);

            // train and corss validate - this returns a model and training meteric for each
            // fold, in this case I have chosen 3 folds
            var cvResults = mlContext.Regression.CrossValidate(transformedData, trainer, numberOfFolds: 3);

            // We will use th RSqured metric to judge the how well we have fitted 
            // the data - the closer to 1, the better the fit.
            // Select all models and use the one with the best RSquared result
            ITransformer[] models =
                cvResults
                    .OrderByDescending(fold => fold.Metrics.RSquared)
                    .Select(fold => fold.Model)
                    .ToArray();

            // Model with best fit
            ITransformer topModel = models[0];

            // print the stats
            Helpers.PrintRegressionFoldsAverageMetrics(trainer.ToString(), cvResults);

            // Save the model for later comsumption from end-user apps
            // since we have 2 pipelines we need to save both
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            if (File.Exists(dataTransformModelPath))
            {
                File.Delete(dataTransformModelPath);
            }

            // save the pipeline used to tranform the data and the pipeline used in training
            using (var file = File.OpenWrite(dataTransformModelPath))
                mlContext.Model.Save(transformer, trainingDataView.Schema, file);

            using (var file = File.OpenWrite(outputModelPath))
                mlContext.Model.Save(topModel, transformedData.Schema, file);
        }
    }
}