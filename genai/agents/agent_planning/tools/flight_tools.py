"""
Flight search and booking tools.

Provides simulated flight search and booking capabilities.
"""

from typing import Dict, Any, List
from datetime import datetime
from .base import BaseTool
from .utils import (
    simulate_latency,
    random_id,
    random_price,
    random_time,
    random_confirmation_code,
    AIRLINES
)
import random


class FlightSearchAPI(BaseTool):
    """Search for available flights between cities."""

    def __init__(self):
        super().__init__(
            name="flight_search_api",
            description="Search for flights between two cities on a specific date. Returns a list of available flights with prices, times, and airlines."
        )

    def _execute(self, origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """
        Search for flights (simulated).

        Args:
            origin: Departure city code (e.g., "NYC", "LAX")
            destination: Arrival city code
            date: Travel date (YYYY-MM-DD format)

        Returns:
            List of flight options sorted by price
        """
        # Simulate API latency
        simulate_latency(150, 350)

        # Generate random number of flight options
        num_flights = random.randint(3, 8)
        flights = []

        for i in range(num_flights):
            departure_time = random_time()
            # Calculate arrival time (add 2-8 hours)
            dep_hour, dep_min = map(int, departure_time.split(':'))
            duration_hours = random.randint(2, 8)
            arr_hour = (dep_hour + duration_hours) % 24
            arrival_time = f"{arr_hour:02d}:{dep_min:02d}"

            flights.append({
                "flight_id": random_id("FL"),
                "airline": random.choice(AIRLINES),
                "origin": origin,
                "destination": destination,
                "date": date,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "price": random_price(150, 800),
                "currency": "USD",
                "duration_minutes": duration_hours * 60,
                "available_seats": random.randint(5, 50),
                "class": random.choice(["Economy", "Business", "First"])
            })

        # Sort by price
        return sorted(flights, key=lambda x: x["price"])

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Departure city or airport code (e.g., 'NYC', 'LAX', 'ORD')"
                },
                "destination": {
                    "type": "string",
                    "description": "Arrival city or airport code (e.g., 'SFO', 'MIA', 'SEA')"
                },
                "date": {
                    "type": "string",
                    "description": "Travel date in YYYY-MM-DD format (e.g., '2024-12-25')"
                }
            },
            "required": ["origin", "destination", "date"]
        }


class BookingAPI(BaseTool):
    """Book a flight using a flight ID."""

    def __init__(self):
        super().__init__(
            name="booking_api",
            description="Book a specific flight using its flight ID. Requires the flight ID from search results and passenger name. Returns booking confirmation."
        )

    def _execute(self, flight_id: str, passenger_name: str, passenger_email: str = None) -> Dict[str, Any]:
        """
        Book a flight (simulated).

        Args:
            flight_id: Flight identifier from search results
            passenger_name: Name for the reservation
            passenger_email: Optional email for confirmation

        Returns:
            Booking confirmation details
        """
        # Simulate booking processing time
        simulate_latency(200, 500)

        return {
            "booking_id": random_id("BK", 5),
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "passenger_email": passenger_email,
            "status": "confirmed",
            "confirmation_code": random_confirmation_code(),
            "booking_date": datetime.now().isoformat(),
            "payment_required": random_price(150, 800),
            "currency": "USD"
        }

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "flight_id": {
                    "type": "string",
                    "description": "Flight ID to book (from search results, e.g., 'FL1234')"
                },
                "passenger_name": {
                    "type": "string",
                    "description": "Full name of passenger for the booking"
                },
                "passenger_email": {
                    "type": "string",
                    "description": "Email address for booking confirmation (optional)"
                }
            },
            "required": ["flight_id", "passenger_name"]
        }
