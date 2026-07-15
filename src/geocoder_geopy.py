
import time

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

geocoding_agent = Nominatim(user_agent="lalitutwal")

def geocode_text_block(address_text: str):
    query_string: str = address_text.strip()
    if not query_string:
        raise ValueError("Empty text block.")

    # written in the documentation of the nominatim
    time.sleep(1)

    try:
        the_result = geocoding_agent.geocode(query_string)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
        return {
            "status": f"ERROR: {type(e).__name__}",
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
        }

    if not the_result:
        return {
            "status": "ZERO_RESULTS",
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
        }

    return the_result


if __name__ == "__main__":
    result = geocode_text_block("1600 Amphitheatre Parkway, Mountain View, CA")
    print(result)
    type(result)
