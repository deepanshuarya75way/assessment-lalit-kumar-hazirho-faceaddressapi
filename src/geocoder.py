import os

import googlemaps
from dotenv import load_dotenv
from googlemaps.geocoding import geocode  # <--- Import the function directly

load_dotenv()
api_key = os.getenv("MAPS_API_KEY")

if not api_key:
    raise ValueError("Missing MAPS_API_KEY")

gmaps = googlemaps.Client(key=api_key)


def geocode_text_block(address_text: str) -> dict:
    query_string = address_text.strip()
    if not query_string:
        raise ValueError("Empty text block.")

    try:
        # Pass the gmaps client directly into the functional call
        geocode_result = geocode(gmaps, query_string)

        if not geocode_result:
            return {
                "status": "ZERO_RESULTS",
                "latitude": None,
                "longitude": None,
                "formatted_address": None,
            }

        primary_match = geocode_result[0]
        coordinates = primary_match["geometry"]["location"]

        return {
            "status": "OK",
            "latitude": float(coordinates["lat"]),
            "longitude": float(coordinates["lng"]),
            "formatted_address": str(primary_match.get("formatted_address", "")),
        }
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        raise e
