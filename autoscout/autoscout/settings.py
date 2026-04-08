BOT_NAME = "autoscout"
SPIDER_MODULES = ["autoscout.spiders"]
NEWSPIDER_MODULE = "autoscout.spiders"

# Splash integration
SPLASH_URL = "http://localhost:8050"

DOWNLOADER_MIDDLEWARES = {
    "scrapy_splash.SplashCookiesMiddleware": 723,
    "scrapy_splash.SplashMiddleware": 725,
    "scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware": 810,
}

SPIDER_MIDDLEWARES = {
    "scrapy_splash.SplashDeduplicateArgsMiddleware": 100,
}

DUPEFILTER_CLASS = "scrapy_splash.SplashAwareDupeFilter"
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"

# Politeness — don't hammer the server
DOWNLOAD_DELAY = 3          # seconds between requests
RANDOMIZE_DOWNLOAD_DELAY = True
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 15
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

CONCURRENT_REQUESTS = 1     # one request at a time to be safe
ROBOTSTXT_OBEY = False      # AutoScout blocks scrapers in robots.txt

# Output
FEEDS = {
    "cars_dataset.csv": {
        "format": "csv",
        "overwrite": True,
    }
}

ITEM_PIPELINES = {
    "autoscout.pipelines.AutoscoutPipeline": 300,
}

# Fake a real browser
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

LOG_LEVEL = "INFO"