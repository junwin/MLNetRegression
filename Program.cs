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
            // Create a ML.NET environment
            MLContext mlContext = new MLContext(seed: 0);

            // Defines the mappig to load text from the input .csv
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

            // Call the method to train the regression

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

            /*
                        var pipeline1 = mlContext.Transforms.Normalize("SoldPrice");
                        var m = pipeline1.Fit(dataView);
                        var modelParams = m.Columns .First(x => x.Name == "SoldPrice")
                                                     .ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;

                        Console.WriteLine($"The normalization parameters are: Scale = {modelParams.Scale} and Offset = {modelParams.Offset}");
                        */

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
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            // </Snippet21>

            //Prediction test
            // Create prediction function and make prediction.
            // <Snippet22>
            var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);
            //var predictionFunction = loadedModel.CreatePredictionEngine<List<HouseData>, List<HousePrediction>>(mlContext);
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
            Console.WriteLine($"Predicted fare: {pv:0.####}, actual fare: 15.5");
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

            var prediction1 = predictionFunction.Predict(housePriceSample);

            var pv1 = prediction1.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {pv1:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

            // 7	3	0	3	0	3
        }
    }
}