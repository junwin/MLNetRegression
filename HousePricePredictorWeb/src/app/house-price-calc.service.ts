import { Injectable } from '@angular/core';
import { HouseFeatures } from './houseFeatures';
import { AREAS } from './mock-areas';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { THROW_IF_NOT_FOUND } from '@angular/core/src/di/injector';

@Injectable({
  providedIn: 'root'
})
export class HousePriceCalcService {
  //api/HousePrice/price?area=76&sqft=1300&rooms=7&bedrooms=3&fullbath=2&halfbath=0&garageType=Attached&garageSpaces=2
  //area=76&houseType=condo&rooms=7&bedRooms=3&bedRoomsBsmt=0&fullBath=2&halfBath=0&approxSquFeet=1200&garageType=attached&garageSpaces=2&parkingSpaces=0
  private housePriceUrl = 'https://localhost:44327/api/HousePrice';  // URL to web api
  private housePriceRootUrl = 'https://localhost:44327/api/HousePrice/price?';  // URL to web api
  predictedPrice: string;

  constructor(private http: HttpClient) { }

  getAreas(): string[] {
    return AREAS;
  }

  getPrice(features: HouseFeatures): Observable<string> {
    var url = this.buildUrl(features);
    var zz = this.http.get<string>(url);
    //zz.subscribe({next(price) {this.predictedPrice = price}})
    //setTimeout(() => { zz. .unsubscribe(); }, 10000);
    return zz;
  }



  buildUrl(features: HouseFeatures): string {
    /*
    var params = {
      parameter1: 'value_1',
      parameter2: 'value 2',
      parameter3: 'value&3'
    };
    */

    var esc = encodeURIComponent;
    var query = Object.keys(features)
      .map(k => esc(k) + '=' + esc(features[k]))
      .join('&');

      var url = this.housePriceRootUrl+query;

     console.log(url) ;
     return url;
  }




}
