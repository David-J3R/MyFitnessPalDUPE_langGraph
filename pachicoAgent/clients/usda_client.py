import json
import sqlite3

import requests

from pachicoAgent.config import config


class USDAClient:
    def __init__(self):
        self.api_key = config.USDA_API_KEY
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.core_nutrients_ids = {
            1003: "Protein",
            1004: "Total lipid (fat)",
            1005: "Carbohydrate, by difference",
            1008: "Energy",
        }

        # Initialize cache
        self.cache_db = "food_cache.db"
        self._init_cache()

    # ----- SQLite Caching Mechanism ----- #
    # Creating a SQLite cache to avoid latency on repeated food details requests
    def _init_cache(self):
        """Creates a tiny SQLite DB to cache food details."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS food_details (
                fdc_id INTEGER PRIMARY KEY, 
                data TEXT)"""
        )
        conn.commit()
        conn.close()

    def _get_from_cache(self, fdc_id: int):
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM food_details WHERE fdc_id = ?", (fdc_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])

    def _save_to_cache(self, fdc_id: int, data: dict):
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO food_details (fdc_id, data) VALUES (?, ?)",
            (fdc_id, json.dumps(data)),
        )
        conn.commit()
        conn.close()

    # ----- USDA API Methods ----- #

    def search_food(self, query: str, limit: int = 5):
        """
        Searches for food items. Prioritizes 'Foundation' and 'Survey' data
        to avoid generic branded duplicates.
        """
        url = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": limit,
            # We strictly want standard reference foods, not random brands
            "dataType": ["Foundation", "Survey (FNDDS)"],
        }
        # Data from USDA comes in 100g portions by default
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}

        data = response.json()
        results = []

        for item in data.get("foods", []):
            # Extract standard nutrients (Proteins, Fats, Carbs, Calories)
            nutrients = {
                n["nutrientName"]: {"value": n["value"], "unit": n["unitName"]}
                for n in item.get("foodNutrients", [])
                if n["nutrientId"] in self.core_nutrients_ids
            }  # IDs for Protein, Fat, Carbs, Energy

            results.append(
                {
                    "fdc_id": item.get("fdcId"),
                    "description": item.get("description"),
                    "brand": item.get("brandOwner", "Generic"),
                    "nutrients": nutrients,
                }
            )

        return results

    # Get food details by portion size
    def get_food_details(self, fdc_id: int):
        """Fetches portion size (weights) for a specific food ID.
        Example: returns that '1 cup' = 240g for a given food item.
        """
        # Check cache first
        cached_data = self._get_from_cache(fdc_id)
        if cached_data:
            print("⚡ Loaded from cache")
            return cached_data

        # Fetch from USDA API
        url = f"{self.base_url}/food/{fdc_id}"
        params = {"api_key": self.api_key}

        # Due to the high latency of USDA API, we add timeout
        try:
            response = requests.get(url, params=params, timeout=10)
        except requests.exceptions.Timeout:
            return {"error": "USDA API request timed out."}

        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}

        data = response.json()

        # Extract portion sizes
        portions = []
        for p in data.get("foodPortions", []):
            # Sometimes it's a measureUnit (cup), sometimes a modifier (slice)
            measure = p.get("measureUnit", {}).get("name") or p.get("modifier")
            amount = p.get("amount", 1)
            gram_weight = p.get("gramWeight")

            # Only include if we have a gram weight
            if gram_weight:
                portions.append(
                    {"label": f"{amount} {measure}", "gram_weight": gram_weight}
                )

        results = {
            "fdc_id": data.get("fdcId"),
            "description": data.get("description"),
            "portions": portions,
        }

        # Save to cache
        self._save_to_cache(fdc_id, results)

        return results


# Simple test block
if __name__ == "__main__":
    client = USDAClient()
    # print(client.search_food("Salmon"))
    """
    FDC ID examples:
    2709223 - Avocado, raw
    2705964 - Chicken breast, rotisserie
    747997 - Eggs, Grade A, Large
    2706287 - Fish salmon, grilled
    """
    print(client.get_food_details(2705964))  # FDC ID
