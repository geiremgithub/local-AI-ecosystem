import json
import urllib.parse
import urllib.request
from datetime import datetime


STAVANGER_NAME = "Stavanger"


def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_stavanger():
    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode(
            {
                "name": STAVANGER_NAME,
                "count": 1,
                "language": "no",
                "format": "json",
            }
        )
    )
    data = fetch_json(url)
    results = data.get("results") or []
    if not results:
        raise RuntimeError("Fant ingen sted som matcher Stavanger i Open-Meteo.")

    place = results[0]
    return {
        "name": place.get("name", STAVANGER_NAME),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": place.get("timezone", "Europe/Oslo"),
    }


def weather_code_to_text(code):
    mapping = {
        0: "Klar himmel",
        1: "Nesten klart",
        2: "Delvis skyet",
        3: "Skyet",
        45: "Tåke",
        48: "Tåke med rimfrost",
        51: "Lette regnbyger",
        53: "Middels regnbyger",
        55: "Kraftige regnbyger",
        56: "Lette isbyger",
        57: "Middels isbyger",
        61: "Lett regn",
        63: "Regn",
        65: "Kraftig regn",
        66: "Lett regn",
        67: "Kraftig regn",
        71: "Lett snø",
        73: "Snø",
        75: "Kraftig snø",
        77: "Snøkorn",
        80: "Lette regnbyger",
        81: "Regnbyger",
        82: "Kraftige regnbyger",
        85: "Lette snøbyger",
        86: "Kraftige snøbyger",
        95: "Torden",
        96: "Torden med hagl",
        99: "Torden med hagl",
    }
    return mapping.get(code, "Ukjent værtype")


def format_day(date_text):
    date_obj = datetime.fromisoformat(date_text)
    return date_obj.strftime("%a %d. %b")


def get_forecast(latitude, longitude, timezone):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max"
        ),
        "forecast_days": 7,
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    return fetch_json(url)


def summarize_rain(forecast_data):
    daily = forecast_data.get("daily") or {}
    dates = daily.get("time") or []
    precipitation = daily.get("precipitation_sum") or [0] * len(dates)
    rain_probability = daily.get("precipitation_probability_max") or [0] * len(dates)
    weather_codes = daily.get("weather_code") or [0] * len(dates)

    rainy_days = []
    for index, day in enumerate(dates):
        if precipitation[index] > 0 or rain_probability[index] >= 50:
            rainy_days.append(
                {
                    "date": day,
                    "precipitation": precipitation[index],
                    "probability": rain_probability[index],
                    "weather_code": weather_codes[index],
                }
            )

    total_precipitation = round(sum(precipitation), 1)
    highest_rain_day = max(rainy_days, key=lambda item: item["precipitation"], default=None)
    most_likely_rain = max(rainy_days, key=lambda item: item["probability"], default=None)

    if not rainy_days:
        return {
            "has_rain": False,
            "summary": "Det ser ut til å bli tørre dager i Stavanger de neste dagene.",
            "total_precipitation": total_precipitation,
            "rainy_days": [],
            "highest_rain_day": None,
            "most_likely_rain": None,
        }

    return {
        "has_rain": True,
        "summary": (
            f"Det ser ut til å bli nedbør i Stavanger de neste dagene. "
            f"Totalt forventes {total_precipitation} mm regn i løpet av perioden."
        ),
        "total_precipitation": total_precipitation,
        "rainy_days": rainy_days,
        "highest_rain_day": highest_rain_day,
        "most_likely_rain": most_likely_rain,
    }


def print_report(place, forecast_data):
    daily = forecast_data.get("daily") or {}
    dates = daily.get("time") or []
    temps_max = daily.get("temperature_2m_max") or [None] * len(dates)
    temps_min = daily.get("temperature_2m_min") or [None] * len(dates)
    precipitation = daily.get("precipitation_sum") or [0] * len(dates)
    rain_probability = daily.get("precipitation_probability_max") or [0] * len(dates)
    weather_codes = daily.get("weather_code") or [0] * len(dates)

    print("=" * 72)
    print(f"Værrapport for {place['name']} ({place['latitude']}, {place['longitude']})")
    print("=" * 72)

    for i, date_text in enumerate(dates):
        day_name = format_day(date_text)
        max_temp = temps_max[i]
        min_temp = temps_min[i]
        precip = precipitation[i]
        probability = rain_probability[i]
        weather_code = weather_codes[i]
        condition = weather_code_to_text(weather_code)

        if max_temp is None or min_temp is None:
            temp_text = "Temperatur ukjent"
        else:
            temp_text = f"{min_temp:.0f}–{max_temp:.0f} °C"

        print(
            f"{day_name:12} | {temp_text:15} | {condition:<20} | "
            f"Nedbør: {precip:.1f} mm | Sannsynlighet: {probability}%"
        )

    print("-" * 72)
    rain_summary = summarize_rain(forecast_data)
    print("Nedbørsanalyse:")
    print(f"- {rain_summary['summary']}")

    if rain_summary["highest_rain_day"]:
        day = rain_summary["highest_rain_day"]
        label = format_day(day["date"])
        print(f"- Mest nedbør: {label} med {day['precipitation']:.1f} mm")

    if rain_summary["most_likely_rain"]:
        day = rain_summary["most_likely_rain"]
        label = format_day(day["date"])
        print(f"- Mest sannsynlig regn: {label} med {day['probability']}% sannsynlighet")

    print("=" * 72)


def main():
    try:
        place = geocode_stavanger()
        forecast_data = get_forecast(place["latitude"], place["longitude"], place["timezone"])
        print_report(place, forecast_data)
    except Exception as exc:
        print("Kunne ikke hente værdata.")
        print(f"Feil: {exc}")
        print("Sjekk internettforbindelsen eller API-et som brukes.")


if __name__ == "__main__":
    main()
