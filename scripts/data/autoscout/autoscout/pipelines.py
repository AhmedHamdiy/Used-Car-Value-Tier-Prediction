import logging
import re

logger = logging.getLogger(__name__)


class AutoscoutPipeline:

    def process_item(self, item, spider):
        # Strip whitespace from all string fields
        for field in item.fields:
            if field in item and isinstance(item[field], str):
                item[field] = item[field].strip()

        # Clean price: "€ 35,950" → "35950"
        if "price" in item and item["price"]:
            item["price"] = "".join(re.findall(r"\d+", item["price"]))

        # Clean mileage: "30,140 km" → "30140"
        if "mileage" in item and item["mileage"]:
            item["mileage"] = "".join(re.findall(r"\d+", item["mileage"]))
        s = f"""Processed: {item.get('brand')} {item.get('model')}
         — €{item.get('price')}"""
        logger.debug(s)
        return item
