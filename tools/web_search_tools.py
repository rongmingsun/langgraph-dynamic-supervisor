"""Tools available to the web search agent (agents/web_search_agent.yaml).

Real HTTP calls to free, keyless services:
  - Wikipedia's MediaWiki Action API (search) + REST API (summary) for
    general web search
  - OpenStreetMap Nominatim for geocoding (a free stand-in for the Google
    Maps Geocoding API, which needs a billed key)
  - OpenStreetMap Overpass API for nearby-place search (a free stand-in
    for the Google Maps Places API) - Nominatim itself only geocodes named
    addresses, it can't answer category queries like "coffee shop near X"
"""

from typing import Optional, Tuple

import requests
from langchain_core.tools import tool

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Nominatim's usage policy requires a descriptive User-Agent identifying the app.
HEADERS = {"User-Agent": "langgraph-web-search-example/1.0"}

# Common place categories mapped to their OpenStreetMap tag. If a query
# doesn't match one of these, find_nearby_places() falls back to matching
# the query against place names directly.
CATEGORY_TAGS = {
    "coffee": ("amenity", "cafe"),
    "cafe": ("amenity", "cafe"),
    "restaurant": ("amenity", "restaurant"),
    "hotel": ("tourism", "hotel"),
    "pharmacy": ("amenity", "pharmacy"),
    "hospital": ("amenity", "hospital"),
    "bank": ("amenity", "bank"),
    "atm": ("amenity", "atm"),
    "gas station": ("amenity", "fuel"),
    "fuel": ("amenity", "fuel"),
    "supermarket": ("shop", "supermarket"),
    "grocery": ("shop", "supermarket"),
    "parking": ("amenity", "parking"),
    "bar": ("amenity", "bar"),
    "park": ("leisure", "park"),
    "school": ("amenity", "school"),
    "bakery": ("shop", "bakery"),
}


@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""
    try:
        search_resp = requests.get(
            WIKIPEDIA_SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
    except requests.RequestException as e:
        return f"[error] Web search failed for '{query}': {e}"

    if not results:
        return f"No results found for: {query}"

    title = results[0]["title"]

    try:
        summary_resp = requests.get(
            WIKIPEDIA_SUMMARY_URL + requests.utils.quote(title, safe=""),
            headers=HEADERS,
            timeout=10,
        )
        summary_resp.raise_for_status()
        summary = summary_resp.json()
    except requests.RequestException as e:
        return f"[error] Failed to fetch summary for '{title}': {e}"

    extract = summary.get("extract", "No summary available.")
    page_url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"{extract} (source: {page_url or f'https://en.wikipedia.org/wiki/{title}'})"


@tool
def geocode_address(address: str) -> str:
    """Look up the latitude/longitude and normalized name for an address or place."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": address, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        return f"[error] Geocoding failed for '{address}': {e}"

    if not results:
        return f"No location found for: {address}"

    top = results[0]
    return f"{top['display_name']} (lat={top['lat']}, lon={top['lon']})"


def _category_tag(query: str) -> Optional[Tuple[str, str]]:
    q = query.lower()
    for keyword, tag in CATEGORY_TAGS.items():
        if keyword in q:
            return tag
    return None


@tool
def find_nearby_places(query: str, near: str) -> str:
    """Find places matching a query (e.g. 'coffee shop') near a given address or place."""
    try:
        anchor_resp = requests.get(
            NOMINATIM_URL,
            params={"q": near, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        anchor_resp.raise_for_status()
        anchor_results = anchor_resp.json()
    except requests.RequestException as e:
        return f"[error] Could not locate '{near}': {e}"

    if not anchor_results:
        return f"No location found for: {near}"

    lat, lon = anchor_results[0]["lat"], anchor_results[0]["lon"]

    tag = _category_tag(query)
    if tag:
        key, value = tag
        place_filter = f'["{key}"="{value}"]'
    else:
        escaped_query = query.replace('"', "")
        place_filter = f'["name"~"{escaped_query}",i]'

    overpass_query = (
        f"[out:json][timeout:15];"
        f"node{place_filter}(around:2000,{lat},{lon});"
        f"out body 5;"
    )

    try:
        places_resp = requests.post(
            OVERPASS_URL,
            data={"data": overpass_query},
            headers=HEADERS,
            timeout=20,
        )
        places_resp.raise_for_status()
        elements = places_resp.json().get("elements", [])
    except requests.RequestException as e:
        return f"[error] Nearby search failed for '{query}' near '{near}': {e}"

    if not elements:
        return f"No results for '{query}' near {near}"

    lines = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed")
        addr = " ".join(
            p for p in (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")) if p
        )
        lines.append(f"- {name}" + (f" ({addr})" if addr else ""))
    return "\n".join(lines)
