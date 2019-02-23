using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace myApp
{
    internal class Program
    {
        
        // Define input files and where the trained model will be stored
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "HouseDataExtended3Anon.csv");

        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "HouseDataExtended3AnonTest.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "housePriceModel.zip");
    

        private static void Main(string[] args)
        {
            // STEP 2: Create a ML.NET environment
            MLContext mlContext = new MLContext(seed: 0);

            HousePriceModel.TrainAndSaveModel(mlContext, _trainDataPath, _modelPath);

            var housePriceSample1 = new HouseData() { Area = 33, BedRooms = 3, BedRoomsBsmt = 0, FullBath = 2, HalfBath = 0, Rooms = 6, Floors = 1, LotSize = 12000, GarageType = "Attached" };
            HousePricePrediction.PredictSinglePrice(housePriceSample1, mlContext, _trainDataPath, _modelPath);

            var housePriceSample2 = new HouseData() { Area = 11, BedRooms = 5, BedRoomsBsmt = 1, FullBath = 6, HalfBath = 1, Rooms = 14, Floors = 2, LotSize = 30000, GarageType = "Attached" };
            HousePricePrediction.PredictSinglePrice(housePriceSample2, mlContext, _trainDataPath, _modelPath);
        }
    }
}