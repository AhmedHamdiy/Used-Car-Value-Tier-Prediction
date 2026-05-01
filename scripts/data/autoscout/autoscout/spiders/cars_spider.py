import scrapy
import json
from scrapy_splash import SplashRequest
from scripts.data.autoscout.autoscout.items import CarItem

# Lua script: tells Splash to wait until JS has rendered the listing items
LISTING_SCRIPT = """
function main(splash, args)
    splash:set_user_agent(args.ua)
    assert(splash:go(args.url))
    splash:wait(3)

    return {html = splash:html()}
end
"""

# Lua script: tells Splash to wait until the price element is visible
DETAIL_SCRIPT = """
function main(splash, args)
    splash:set_user_agent(args.ua)
    assert(splash:go(args.url))
    splash:wait(2)

    return {html = splash:html()}
end
"""

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


class CarsSpider(scrapy.Spider):
    name = "cars"
    base_url = "https://www.autoscout24.com"
    start_page = 1
    total_pages = 200  # ← adjust as needed
    use_splash_for_listings = True

    custom_settings = {
        # Faster, still controlled
        "CONCURRENT_REQUESTS": 12,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 6,
        "DOWNLOAD_DELAY": 0.25,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 3.0,
    }

    def _request_headers(self):
        return {
            "User-Agent": UA,
            "Accept-Language": "en-US,en;q=0.9",
        }

    def start_requests(self):
        """Kick off by requesting listing pages (Splash for listing,
        plain HTTP for details)."""
        for page in range(self.start_page, self.total_pages + 1):
            url = f"{self.base_url}/lst?page={page}"
            if self.use_splash_for_listings:
                yield SplashRequest(
                    url=url,
                    callback=self.parse_listing,
                    endpoint="execute",
                    args={
                        "lua_source": LISTING_SCRIPT,
                        "ua": UA,
                        "timeout": 30,
                        "max_wait": 10,
                    },
                    meta={"page": page},
                )
            else:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_listing,
                    headers=self._request_headers(),
                    meta={"page": page},
                )

    # ---------------------------------------------------
    # STEP 1: Parse a listing page → extract car links
    # ---------------------------------------------------
    def parse_listing(self, response):
        page = response.meta["page"]

        # Primary strategy: parse Next.js hydrated payload (fast + stable)
        links = []
        next_data_text = response.css("script#__NEXT_DATA__::text").get()
        if next_data_text:
            try:
                next_data = json.loads(next_data_text)
                listings = (
                    next_data.get("props", {}).get("pageProps", {})
                    .get("listings", [])
                )
                links = [
                    listing.get("url", "")
                    for listing in listings
                    if isinstance(listing, dict) and listing.get("url")
                ]
            except json.JSONDecodeError:
                self.logger.warning("Page %s: could not decode"
                                    "__NEXT_DATA__.", page)

        # Fallback strategy: parse rendered anchors
        if not links:
            css_sel = 'a[href*="/offers/"]::attr(href)'
            links = response.css(css_sel).getall()

        # De-duplicate while preserving order
        links = list(dict.fromkeys(links))

        self.logger.info(f"Page {page}: found {len(links)} car link(s).")

        for href in links:
            if href.startswith("http"):
                full_url = href
            else:
                full_url = self.base_url + href
            # Detail pages already contain required fields in HTML,
            # So normal Request is much faster.
            yield scrapy.Request(
                url=full_url,
                callback=self.parse_car,
                headers=self._request_headers(),
                meta={"url": full_url},
            )

    # -------------------------------------------------------
    # STEP 2: Parse a car detail page → extract all 10 fields
    # -------------------------------------------------------
    def parse_car(self, response):
        item = CarItem()
        item["url"] = response.meta["url"]

        # ── Brand ──────────────────────────────────────────
        item["brand"] = (
            response.css("span.StageTitle_boldClassifiedInfo__sQb0l::text")
            .get(default="")
            .strip()
            .split()[0]
        )

        # ── Model ──────────────────────────────────────────
        model_parts = (
            response.css("span.StageTitle_boldClassifiedInfo__sQb0l::text")
            .get(default="")
            .strip()
            .split()[1:]
        )
        item["model"] = " ".join(model_parts)

        # ── Price ──────────────────────────────────────────
        # Remove the superscript footnote before grabbing text
        item["price"] = (
            response.css("span.PriceInfo_price__XU0aF::text")
            .get(default="").strip()
        )

        # ── Overview quick-facts:
        # (mileage, gearbox, power, fuel, year, seller)
        item["mileage"] = self._get_overview(response, "Mileage")
        item["gearbox"] = self._get_overview(response, "Gearbox")
        item["power"] = self._get_overview(response, "Power")
        item["fuelType"] = self._get_overview(response, "Fuel type")
        item["year"] = self._get_overview(response, "First registration")
        item["seller"] = self._get_overview(response, "Seller")

        # ── Body / vehicle type (from Basic Data detail section) ──
        item["vehicleType"] = self._get_detail_field(response, "Body type")

        # Fallback: get year from detail section if overview missed it
        if not item.get("year"):
            item["year"] = self._get_detail_field(response,
                                                  "First registration")

        yield item

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _get_overview(self, response, label: str) -> str:
        """
        The VehicleOverview grid pairs each title div with a value div.
        We find the title matching `label` then grab
        the next value div's text.
        """
        titles = response.css("div.VehicleOverview_itemTitle__S2_lb")
        for title in titles:
            if title.css("::text").get("").strip().lower() == label.lower():
                # The value div is the next sibling
                # after the empty separator div
                container = title.xpath("parent::div")
                value = (
                    container.css("div.VehicleOverview_itemText__AI4dA::text")
                    .get(default="")
                    .strip()
                )
                return value
        return ""

    def _get_detail_field(self, response, label: str) -> str:
        """
        In the Basic Data section,
            fields are <dt>label</dt><dd>value</dd> pairs.
        """
        dts = response.css("dt")
        for dt in dts:
            if dt.css("::text").get("").strip().lower() == label.lower():
                dd = dt.xpath("following-sibling::dd[1]")
                return dd.css("::text").get(default="").strip()
        return ""
