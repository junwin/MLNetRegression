using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;

namespace HousePriceService.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HousePriceController : ControllerBase
    {
        // GET api/HousePrice
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            var housePriceSample1 = new HouseData() { Area = "76", BedRooms = 3, BedRoomsBsmt = 0, FullBath = 2, HalfBath = 0, Rooms = 7, ApproxSquFeet = 1300, GarageType = "Attached", GarageSpaces = 2 };
            //var price = HousePricePrediction.PredictSinglePrice(housePriceSample1, @"MLNETModels\housePriceModel.zip");
            var price = HousePricePrediction.PredictSinglePrice(housePriceSample1);

            return new string[] { price };
        }

        // GET api/HousePrice/price
        [HttpGet("{price}")]
        public ActionResult<IEnumerable<string>> Get(string area, float approxSquFeet, float rooms, float bedrooms, float fullbath, float halfbath, string garageType, float garageSpaces)
        {
            var housePriceSample1 = new HouseData() { Area = area, BedRooms = bedrooms, BedRoomsBsmt = 0, FullBath = fullbath, HalfBath =halfbath, Rooms = rooms, ApproxSquFeet = approxSquFeet, GarageType = garageType, GarageSpaces = garageSpaces };
            var price = HousePricePrediction.PredictSinglePrice(housePriceSample1);
            return new string[] { price };
            
            
        }
    }
}