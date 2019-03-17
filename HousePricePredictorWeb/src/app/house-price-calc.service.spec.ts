import { TestBed } from '@angular/core/testing';

import { HousePriceCalcService } from './house-price-calc.service';

describe('HousePriceCalcService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: HousePriceCalcService = TestBed.get(HousePriceCalcService);
    expect(service).toBeTruthy();
  });
});
