using Microsoft.ML;
using System;
using System.IO;

namespace myApp
{
    internal class Program
    {
        // Define input files and where the trained model will be stored
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "HouseData.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "housePriceModel.zip");

        private static void Main(string[] args)
        {
            // STEP 2: Create a ML.NET environment
            MLContext mlContext = new MLContext(seed: 0);

            // Train and Save the model
            HousePriceModel.TrainAndSaveModel(mlContext, _trainDataPath, _modelPath);

            // Run a few test examples
            var housePriceSample1 = new HouseData() { Area = "76", BedRooms = 3, BedRoomsBsmt = 0, FullBath = 2, HalfBath = 0, Rooms = 7, ApproxSquFeet=1300, GarageType = "Attached", GarageSpaces=2 };
            HousePricePrediction.PredictSinglePrice(housePriceSample1, mlContext, _trainDataPath, _modelPath);

            var housePriceSample2 = new HouseData() { Area = "62", BedRooms = 5, BedRoomsBsmt = 1, FullBath = 6, HalfBath = 1, Rooms = 16, ApproxSquFeet = 8410, GarageType = "Attached", GarageSpaces = 4 };
            HousePricePrediction.PredictSinglePrice(housePriceSample2, mlContext, _trainDataPath, _modelPath);

        }
    }
}