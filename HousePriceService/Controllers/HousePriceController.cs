using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;

namespace HousePriceService.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HousePriceController : ControllerBase
    {
        // For testing point to the place where the trainer had out put the models
        private string _modelPath = @"C:\Users\junwi\source\repos\MLNetRegression\HousePriceTraining\bin\Debug\netcoreapp2.2\housePriceModel.zip";
        private string _dataTransformModelPath = @"C:\Users\junwi\source\repos\MLNetRegression\HousePriceTraining\bin\Debug\netcoreapp2.2\housePriceDataTransformer.zip";

        // GET api/HousePrice
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            var housePriceSample1 = new HouseData() { Area = "76", BedRooms = 3, BedRoomsBsmt = 0, FullBath = 2, HalfBath = 0, Rooms = 7, ApproxSquFeet = 1300, GarageType = "Attached", GarageSpaces = 2 };
            HouseData[] hd = new HouseData[] { housePriceSample1 };

            var results = HousePricePrediction.PredictSinglePriceSet(hd, _dataTransformModelPath, _modelPath);

            return new string[] { results[0].ToString() };
        }

        // GET api/HousePrice/price
        // GET api/HousePrice/price?area=76&sqft=1300&rooms=7&bedrooms=3&fullbath=2&halfbath=0&garageType=Attached&garageSpaces=2
        //area=76&houseType=condo&rooms=7&bedRooms=3&bedRoomsBsmt=0&fullBath=2&halfBath=0&approxSquFeet=1200&garageType=attached&garageSpaces=2&parkingSpaces=0
        [HttpGet("{price}")]
        public ActionResult<IEnumerable<string>> Get([FromQuery] string area, [FromQuery] float approxSquFeet, [FromQuery] float rooms, [FromQuery] float bedrooms, [FromQuery] float fullbath,
            [FromQuery] float halfbath, [FromQuery] string garageType, [FromQuery] float garageSpaces)
        {
            var requestHousePriceFeatures = new HouseData() { Area = area, BedRooms = bedrooms, BedRoomsBsmt = 0, FullBath = fullbath, HalfBath = halfbath, Rooms = rooms, ApproxSquFeet = approxSquFeet, GarageType = garageType, GarageSpaces = garageSpaces };
            HouseData[] hd = new HouseData[] { requestHousePriceFeatures };

            var results = HousePricePrediction.PredictSinglePriceSet(hd, _dataTransformModelPath, _modelPath);

            return new string[] { results[0].ToString() };
        }
    }
}