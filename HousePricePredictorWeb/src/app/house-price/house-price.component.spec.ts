import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { HousePriceComponent } from './house-price.component';

describe('HousePriceComponent', () => {
  let component: HousePriceComponent;
  let fixture: ComponentFixture<HousePriceComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ HousePriceComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(HousePriceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
