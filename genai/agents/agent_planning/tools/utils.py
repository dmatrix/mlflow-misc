"""
Utility functions for generating realistic fake data for tools.

These utilities help create simulated API responses that look realistic
for the tutorial demonstrations.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


def random_date_in_future(min_days: int = 1, max_days: int = 90) -> str:
    """
    Generate a random date in the future.

    Args:
        min_days: Minimum days from now
        max_days: Maximum days from now

    Returns:
        Date string in YYYY-MM-DD format
    """
    days = random.randint(min_days, max_days)
    future_date = datetime.now() + timedelta(days=days)
    return future_date.strftime("%Y-%m-%d")


def random_time() -> str:
    """
    Generate a random time in HH:MM format.

    Returns:
        Time string (e.g., "14:30")
    """
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"


def random_datetime(days_offset: int = 0) -> str:
    """
    Generate a random datetime ISO string.

    Args:
        days_offset: Days to offset from now (can be negative)

    Returns:
        ISO datetime string
    """
    base_date = datetime.now() + timedelta(days=days_offset)
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    dt = base_date.replace(hour=random_hour, minute=random_minute, second=0, microsecond=0)
    return dt.isoformat()


def random_id(prefix: str, length: int = 4) -> str:
    """
    Generate a random ID with prefix.

    Args:
        prefix: ID prefix (e.g., "FL" for flight)
        length: Number of random digits

    Returns:
        ID string (e.g., "FL1234")
    """
    max_num = 10 ** length - 1
    min_num = 10 ** (length - 1)
    return f"{prefix}{random.randint(min_num, max_num)}"


def random_price(min_price: int = 50, max_price: int = 1000) -> int:
    """
    Generate a random price.

    Args:
        min_price: Minimum price
        max_price: Maximum price

    Returns:
        Random price as integer
    """
    return random.randint(min_price, max_price)


def random_email() -> str:
    """
    Generate a random email address.

    Returns:
        Email address string
    """
    domains = ["example.com", "test.com", "demo.com", "sample.com"]
    names = ["john", "jane", "alex", "sam", "chris", "pat"]
    return f"{random.choice(names)}{random.randint(1, 999)}@{random.choice(domains)}"


def random_phone() -> str:
    """
    Generate a random phone number.

    Returns:
        Phone number string (US format)
    """
    area_code = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number = random.randint(1000, 9999)
    return f"+1-{area_code}-{exchange}-{number}"


def random_name() -> str:
    """
    Generate a random name.

    Returns:
        Full name string
    """
    first_names = ["John", "Jane", "Alex", "Sam", "Chris", "Pat", "Jordan", "Taylor"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def random_address() -> Dict[str, str]:
    """
    Generate a random address.

    Returns:
        Address dict with street, city, state, zip
    """
    streets = ["Main St", "Oak Ave", "Maple Dr", "Park Blvd", "Lake Rd"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    states = ["NY", "CA", "IL", "TX", "AZ"]

    return {
        "street": f"{random.randint(100, 9999)} {random.choice(streets)}",
        "city": random.choice(cities),
        "state": random.choice(states),
        "zip": f"{random.randint(10000, 99999)}"
    }


def random_confirmation_code() -> str:
    """
    Generate a random confirmation code.

    Returns:
        Confirmation code (e.g., "A1B2C3")
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join([
        random.choice(letters),
        str(random.randint(0, 9)),
        random.choice(letters),
        str(random.randint(0, 9)),
        random.choice(letters),
        str(random.randint(0, 9))
    ])


def simulate_latency(min_ms: float = 100, max_ms: float = 500):
    """
    Simulate API latency with random sleep.

    Args:
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
    """
    import time
    latency = random.uniform(min_ms, max_ms) / 1000
    time.sleep(latency)


# Common data for tools to use
AIRLINES = ["United", "Delta", "American", "Southwest", "JetBlue", "Alaska"]
CITIES = {
    "NYC": "New York",
    "LAX": "Los Angeles",
    "ORD": "Chicago",
    "DFW": "Dallas",
    "DEN": "Denver",
    "SFO": "San Francisco",
    "SEA": "Seattle",
    "MIA": "Miami",
    "BOS": "Boston",
    "ATL": "Atlanta"
}
HOTELS = ["Marriott", "Hilton", "Hyatt", "Holiday Inn", "Best Western", "Sheraton"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Snowy", "Windy"]
