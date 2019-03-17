import { Component, OnInit } from '@angular/core';
import { HouseFeatures } from '../houseFeatures';
import { AREAS } from '../mock-areas';
import { HousePriceCalcService } from '../house-price-calc.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-house-price',
  templateUrl: './house-price.component.html',
  styleUrls: ['./house-price.component.css']
})
export class HousePriceComponent implements OnInit {

  houseFeature: HouseFeatures = {
     area: '76',
    houseType: 'condo',
    rooms: 7,
    bedRooms: 3,
    bedRoomsBsmt: 0,
    fullBath: 2,
    halfBath: 0,
    approxSquFeet: 1200,
    garageType: 'attached',
    garageSpaces: 2,
    parkingSpaces: 0
  };

  availableAreas: string[];
  selectedArea: string;
  housePrice: Observable<string>;

  constructor(private priceCalcService: HousePriceCalcService) { }

  ngOnInit() {
    this.availableAreas = this.priceCalcService.getAreas();
  }



  onSelectArea(a: string): void {
    this.selectedArea = a;
    this.housePrice = this.priceCalcService.getPrice(this.houseFeature);
  }
}


