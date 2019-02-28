using Microsoft.ML.Data;

namespace myApp
{
    public class HouseData
    {
        [LoadColumn(3)]
        public float Area;

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
}