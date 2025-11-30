"""Hotel search tools."""

from typing import Dict, Any, List
from .base import BaseTool
from .utils import simulate_latency, random_id, random_price, HOTELS
import random


class HotelSearchAPI(BaseTool):
    """Search for hotels in a location."""

    def __init__(self):
        super().__init__(
            name="hotel_search_api",
            description="Search for hotels in a specific location with check-in and check-out dates. Returns available hotels with prices and amenities."
        )

    def _execute(self, location: str, check_in: str, check_out: str) -> List[Dict[str, Any]]:
        """Search for hotels (simulated)."""
        simulate_latency(200, 400)

        num_hotels = random.randint(4, 10)
        hotels = []

        for i in range(num_hotels):
            hotels.append({
                "hotel_id": random_id("HTL"),
                "name": f"{random.choice(HOTELS)} {random.choice(['Downtown', 'Airport', 'Convention Center', 'Waterfront'])}",
                "location": location,
                "rating": round(random.uniform(3.0, 5.0), 1),
                "price_per_night": random_price(80, 400),
                "currency": "USD",
                "amenities": random.sample(["WiFi", "Pool", "Gym", "Parking", "Breakfast", "Spa"], k=random.randint(2, 4)),
                "available_rooms": random.randint(1, 10)
            })

        return sorted(hotels, key=lambda x: x["price_per_night"])

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location to search"},
                "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"}
            },
            "required": ["location", "check_in", "check_out"]
        }
