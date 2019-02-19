using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace myApp
{
    internal class Program
    {
        // Mapping the input data
        //StreetNum	StrName	Area	Rooms	FullBath	HalfBath	Type	Zip	SoldPrice	ListPrice	MT	BedsBsmt	Beds	GarageSpaces	ClosedDate	GarageType	BsmtBath	AbvGrdSF	BsmntSF	Style	LotDims	LotSize

        public class HouseData
        {
            [LoadColumn(4)]
            public float Rooms;

            [LoadColumn(13)]
            public float BedRooms;

            [LoadColumn(12)]
            public float BedRoomsBsmt;

            [LoadColumn(5)]
            public float FullBath;

            [LoadColumn(6)]
            public float HalfBath;

            [LoadColumn(7)]
            public float Floors;

            [LoadColumn(9)]
            public float SoldPrice;

            [LoadColumn(22)]
            public float LotSize;

            [LoadColumn(16)]
            public string GarageType;
        }

        //  The output datat
        public class HousePrediction
        {
            [ColumnName("Score")]
            public float SoldPrice;
        }

        // Define input files and where the trained model will be stored
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "HouseDataExtended3Anon.csv");

        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "HouseDataExtended3AnonTest.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Model.zip");
        private static TextLoader _textLoader;

        private static void Main(string[] args)
        {
            // STEP 2: Create a ML.NET environment
            MLContext mlContext = new MLContext(seed: 0);

            // Defines the mappig to load text from the input .csv
            // An improved version would allow the user to choose the 
            // features to be loaded.
            // This would make it easy to see what features are useful from the
            // gamut provided
            _textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Rooms", DataKind.R4, 4),
                    new TextLoader.Column("BedRooms", DataKind.R4, 13),
                    new TextLoader.Column("BedRoomsBsmt", DataKind.R4, 12),
                    new TextLoader.Column("FullBath", DataKind.R4, 5),
                    new TextLoader.Column("HalfBath", DataKind.R4, 6),
                    new TextLoader.Column("Floors", DataKind.R4, 7),
                    new TextLoader.Column("LotSize", DataKind.R4, 22),
                    new TextLoader.Column("GarageType", DataKind.Text, 16),
                    new TextLoader.Column("SoldPrice", DataKind.R4, 9)
                }
            });

            //  Call the method to train the regression
            var model = Train(mlContext, _trainDataPath);

            //  Evaluate the model and compare with the test data set
            Evaluate(mlContext, model);

            //  Execute the model for some sample data, this could be the entry point from
            // some web service, where a realtor wants to get a quick price estimate
            TestSinglePrediction(mlContext);

            Console.WriteLine("Press any key to exit....");
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            // You can use the ML.Net normailzers if needs be.
            /*
              var pipeline1 = mlContext.Transforms.Normalize("SoldPrice");
              var m = pipeline1.Fit(dataView);
              var modelParams = m.Columns .First(x => x.Name == "SoldPrice")
                                                     .ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;

              Console.WriteLine($"The normalization parameters are: Scale = {modelParams.Scale} and Offset = {modelParams.Offset}");
            */

            // THis defines a pipeline used to train the model
            // Note that a good idea would be to vary the number of features and map that agains the 
            // metrics from the test data to avoid over or undefitting
            // A use feature is the ML.Net support for converting string values to a numeric vector that
            // can be used to train.
            var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: "SoldPrice", outputColumnName: "Label")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("GarageType"))
.Append(mlContext.Transforms.Concatenate("Features", "Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "Floors", "GarageType", "LotSize"))
.Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            SaveModelAsFile(mlContext, model);

            return model;
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        /// <summary>
        /// Evaluate the test data set and write metrics to show the model 
        /// quality
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);
            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            //  Load the prediction model we saved earlier
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            

            
            var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);
           
            var housePriceSample = new HouseData()
            {
                BedRooms = 3,
                BedRoomsBsmt = 0,
                FullBath = 2,
                HalfBath = 0,
                Rooms = 6,
                Floors = 1,
                LotSize = 12000,
                GarageType = "Attached"
            };

            var prediction = predictionFunction.Predict(housePriceSample);

            var pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted SellPrice: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");

            housePriceSample = new HouseData()
            {
                BedRooms = 5,
                BedRoomsBsmt = 1,
                FullBath = 6,
                HalfBath = 1,
                Rooms = 14,
                Floors = 2,
                LotSize = 30000,
                GarageType = "Attached"
            };

            prediction = predictionFunction.Predict(housePriceSample);

            pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Sell Price: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");

            // 7	3	0	3	0	3
        }
    }
}