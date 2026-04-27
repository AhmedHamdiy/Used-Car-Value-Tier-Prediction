import scrapy

class CarItem(scrapy.Item):
    url        = scrapy.Field()
    brand      = scrapy.Field()
    model      = scrapy.Field()
    price      = scrapy.Field()
    seller     = scrapy.Field()
    vehicleType = scrapy.Field()
    year       = scrapy.Field()
    gearbox    = scrapy.Field()
    power      = scrapy.Field()
    mileage    = scrapy.Field()
    fuelType   = scrapy.Field()