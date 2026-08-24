import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime


START_ADDRESS = "Laberget 26, Stavanger, Norway"
END_ADDRESS = "Seljeveien 1, Stavanger, Norway"

# Endre dette når du vil teste en annen avreisetid.
# Standard er klokken akkurat nå i lokal tid.
DEPARTURE_TIME = datetime.now().astimezone()


def fetch_json(url, params=None):
    full_url = url
    if params:
        full_url = url + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrafikkAgent/1.0",
            "Accept-Language": "no"
        },
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    data = fetch_json(url, params)

    if not data:
        raise RuntimeError(f"Fant ingen koordinater for adressen: {address}")

    place = data[0]
    return {
        "lat": float(place["lat"]),
        "lon": float(place["lon"]),
        "display_name": place.get("display_name", address),
    }


def get_route(start_coords, end_coords):
    coords = f"{start_coords['lon']},{start_coords['lat']};{end_coords['lon']},{end_coords['lat']}"
    url = f"https://router.project-osrm.org/route/v1/driving/{coords}"
    params = {
        "overview": "false",
        "alternatives": "false",
        "steps": "true",
        "geometries": "geojson",
    }
    data = fetch_json(url, params)

    if "routes" not in data or not data["routes"]:
        raise RuntimeError(f"OSRM returnerte ingen rute for {START_ADDRESS} til {END_ADDRESS}.")

    route = data["routes"][0]
    return {
        "distance_meters": float(route.get("distance", 0)),
        "duration_seconds": float(route.get("duration", 0)),
        "legs": route.get("legs", []),
        "summary": route.get("summary", "Ingen sammenfatning"),
    }


def format_duration(seconds):
    total_minutes = int(round(seconds / 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours > 0:
        return f"{hours} t {minutes} min"
    return f"{minutes} min"


def print_route_report(start_address, end_address, start_coords, end_coords, route, departure_time):
    distance_km = route["distance_meters"] / 1000
    duration_text = format_duration(route["duration_seconds"])
    departure_str = departure_time.strftime("%d.%m.%Y %H:%M")

    print("=" * 80)
    print("Trafikk- og reiserapport")
    print("=" * 80)
    print(f"Fra: {start_address}")
    print(f"Til: {end_address}")
    print(f"Startkoordinater: {start_coords['lat']:.5f}, {start_coords['lon']:.5f}")
    print(f"Målkoordinater:  {end_coords['lat']:.5f}, {end_coords['lon']:.5f}")
    print(f"Avreisetid: {departure_str}")
    print("")
    print(f"Avstand: {distance_km:.1f} km")
    print(f"Reisetid: {duration_text}")
    print(f"Rutesammendrag: {route['summary']}")
    print("")
    print("Trafikkstatus:")
    print("- Bruker det offentlige OSRM-directions-API-et for en live ruteberegning.")
    print("- Gratis OSRM har ikke en detaljert 'trafikkindeks'-modell som premium-tjenester, så dette er rute- og kjøretidsestimert basert på vanlig routingdata.")
    print("")
    print("Etapper:")
    for idx, leg in enumerate(route["legs"], start=1):
        leg_distance = float(leg.get("distance", 0)) / 1000
        leg_duration = format_duration(float(leg.get("duration", 0)))
        print(f"  {idx}. Etappe: {leg_distance:.1f} km, varighet {leg_duration}")

    print("=" * 80)


def main():
    try:
        start_coords = geocode_address(START_ADDRESS)
        end_coords = geocode_address(END_ADDRESS)
        route = get_route(start_coords, end_coords)
        print_route_report(START_ADDRESS, END_ADDRESS, start_coords, end_coords, route, DEPARTURE_TIME)
    except urllib.error.HTTPError as err:
        print("Feil i HTTP-kallet mot åpne tjenester.")
        print(f"Statuskode: {err.code}")
        print(f"Detaljer: {err.reason}")
    except urllib.error.URLError as err:
        print("Nettverksfeil eller API utilgjengelig.")
        print(f"Detaljer: {err.reason}")
    except Exception as err:
        print("Kunne ikke beregne ruten.")
        print(f"Feil: {err}")


if __name__ == "__main__":
    main()
