using Microsoft.ML.Data;

//MLS #	Stat	Closed Date	Sold Pr	Type	Yr Blt	Street #	Str Name	Area	Zip	ASF	# Rms	# Full Baths	# Half Baths	Bsmt Bth	Beds	Bsmt. Beds	Garage Type	# Garage Spaces	# Parking Spaces

namespace myApp
{
    public class HouseData
    {
        [LoadColumn(8)]
        public string Area;

        [LoadColumn(4)]
        public string HouseType;

        [LoadColumn(11)]
        public float Rooms;

        [LoadColumn(15)]
        public float BedRooms;

        [LoadColumn(16)]
        public float BedRoomsBsmt;

        [LoadColumn(12)]
        public float FullBath;

        [LoadColumn(13)]
        public float HalfBath;

        [LoadColumn(3)]
        public float Label;

        [LoadColumn(10)]
        public float ApproxSquFeet;

        [LoadColumn(17)]
        public string GarageType;

        [LoadColumn(18)]
        public float GarageSpaces;

        [LoadColumn(19)]
        public float ParkingSpaces;
    }

    //  The output datat
    public class HousePrediction
    {
        [ColumnName("Score")]
        public float SoldPrice;
    }
}