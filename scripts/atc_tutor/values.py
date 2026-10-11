"""Typed value tables and spoken / plain-language renderers for generated ATC rows.

Sources: ICAO Doc 8585 telephony designators (airline list below), ICAO Annex 10
Vol II / Doc 9432 number pronunciation, FAA JO 7110.65 for US phrasing, EUROCONTROL
8.33 kHz channel naming. Airports, runways and reporting points are generated or
neutral so no unverifiable real-world pairing is asserted.
"""

import random

TELEPHONY = {
    # telephony: airline (ICAO Doc 8585)
    "speedbird": "British Airways", "shamrock": "Aer Lingus", "lufthansa": "Lufthansa",
    "swiss": "Swiss", "edelweiss": "Edelweiss Air", "air france": "Air France",
    "klm": "KLM", "iberia": "Iberia", "alitalia": "ITA Airways", "austrian": "Austrian Airlines",
    "scandinavian": "SAS", "finnair": "Finnair", "tap": "TAP Air Portugal",
    "csa": "Czech Airlines", "smartwings": "Smartwings", "skytravel": "Smartwings",
    "lot": "LOT Polish Airlines", "easy": "easyJet", "ryanair": "Ryanair",
    "wizz air": "Wizz Air", "vueling": "Vueling", "eurowings": "Eurowings",
    "condor": "Condor", "turkish": "Turkish Airlines", "emirates": "Emirates",
    "qatari": "Qatar Airways", "etihad": "Etihad Airways", "saudia": "Saudia",
    "el al": "El Al", "aeroflot": "Aeroflot", "springbok": "South African Airways",
    "kenya": "Kenya Airways", "ethiopian": "Ethiopian Airlines", "egyptair": "EgyptAir",
    "qantas": "Qantas", "jetstar": "Jetstar", "velocity": "Virgin Australia",
    "rex": "Regional Express", "qlink": "QantasLink", "new zealand": "Air New Zealand",
    "singapore": "Singapore Airlines", "cathay": "Cathay Pacific", "air china": "Air China",
    "china southern": "China Southern", "china eastern": "China Eastern", "japan air": "Japan Airlines",
    "all nippon": "All Nippon Airways", "korean air": "Korean Air", "asiana": "Asiana",
    "thai": "Thai Airways", "malaysian": "Malaysia Airlines", "garuda": "Garuda Indonesia",
    "air india": "Air India", "american": "American Airlines", "delta": "Delta Air Lines",
    "united": "United Airlines", "southwest": "Southwest Airlines", "jetblue": "JetBlue",
    "alaska": "Alaska Airlines", "spirit wings": "Spirit Airlines", "frontier flight": "Frontier Airlines",
    "air canada": "Air Canada", "westjet": "WestJet", "aeromexico": "Aeromexico",
    "avianca": "Avianca", "lan chile": "LATAM", "fedex": "FedEx", "ups": "UPS",
    "brickyard": "Republic Airways", "skywest": "SkyWest",
    "dlh": "Lufthansa", "nav checker": "a flight-inspection aircraft",
}
REGISTRATION_PREFIX = {
    "icao_europe": [("hotel bravo", "Switzerland", 3), ("oscar kilo", "Czech Republic", 3),
                    ("oscar mike", "Slovakia", 3), ("delta", "Germany", 4),
                    ("golf", "United Kingdom", 4), ("foxtrot", "France", 4)],
    "icao_australia": [("victor hotel", "Australia", 3)],
    "faa": [("november", "United States", None)],
}
LETTERS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike "
           "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
           "yankee zulu").replace("xray", "x_ray").split()
LETTERS = [letter.replace("_", " ") for letter in LETTERS]
DIGITS = "zero one two three four five six seven eight nine".split()
TEENS = ("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words_int(n):
    """Plain cardinal words: 1008 -> 'one thousand eight', 270 -> 'two hundred seventy'."""
    if n < 10:
        return DIGITS[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + DIGITS[n % 10])
    if n < 1000:
        rest = n % 100
        return DIGITS[n // 100] + " hundred" + ("" if rest == 0 else " " + words_int(rest))
    rest = n % 1000
    return words_int(n // 1000) + " thousand" + ("" if rest == 0 else " " + words_int(rest))


def spoken_digits(text, faa=False):
    """Digit-by-digit radio form; ICAO 'decimal', FAA 'point'."""
    out = []
    for char in text:
        if char.isdigit():
            out.append(DIGITS[int(char)])
        elif char == ".":
            out.append("point" if faa else "decimal")
    return " ".join(out)


def spoken_feet(feet):
    thousands, hundreds = divmod(feet, 1000)
    parts = []
    if thousands:
        parts.append(spoken_digits(str(thousands)) + " thousand")
    if hundreds:
        parts.append(DIGITS[hundreds // 100] + " hundred")
    return " ".join(parts) + " feet"


def runway_words(designator):
    number, side = designator[:2], designator[2:]
    side_word = {"L": " left", "R": " right", "C": " center", "": ""}[side]
    return words_int(int(number)) + side_word


COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def compass(degrees):
    return COMPASS[round((degrees % 360) / 45) % 8]


class Sampler:
    """Seeded sampler of typed slot values with spoken and plain renderings."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def callsign(self, dialect):
        if self.rng.random() < 0.55:
            telephony = self.rng.choice(sorted(TELEPHONY))
            number = self.rng.choice([
                str(self.rng.randint(1, 99)), str(self.rng.randint(100, 999)),
                str(self.rng.randint(1000, 9999)),
                str(self.rng.randint(1, 99)) + self.rng.choice("ABCDEFGHJKLMNPRSTWXYZ"),
            ])
            spoken = telephony + " " + " ".join(
                DIGITS[int(c)] if c.isdigit() else LETTERS[ord(c) - 65] for c in number
            )
            return {"type": "airline", "telephony": telephony, "airline": TELEPHONY[telephony],
                    "number": number, "spoken": spoken}
        prefix, country, length = self.rng.choice(REGISTRATION_PREFIX[dialect])
        if length is None:
            tail = str(self.rng.randint(1, 999)) + "".join(self.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(self.rng.randint(0, 2)))
        else:
            tail = "".join(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))
        spoken = prefix + " " + " ".join(
            DIGITS[int(c)] if c.isdigit() else LETTERS[ord(c) - 65] for c in tail
        )
        return {"type": "registration", "country": country, "tail": tail, "spoken": spoken}

    def abbreviated_callsign(self):
        """Last two or three registration letters only (no prefix): 'papa mike x ray'."""
        tail = "".join(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(self.rng.choice([2, 3, 3])))
        return {"type": "abbreviated", "tail": tail,
                "spoken": " ".join(LETTERS[ord(c) - 65] for c in tail)}

    def low_flight_level(self):
        value = self.rng.choice(range(30, 151, 10))
        return {"value": value, "spoken": "flight level " + spoken_digits(f"{value:03d}"),
                "plain_feet": words_int(value * 100) + " feet"}

    def heading(self):
        value = self.rng.randrange(5, 361, 5)
        return {"value": value, "spoken": "heading " + spoken_digits(f"{value:03d}"),
                "plain": words_int(value) + " degrees", "direction": compass(value)}

    def flight_level(self):
        value = self.rng.choice(range(60, 451, 10))
        return {"value": value, "spoken": "flight level " + spoken_digits(f"{value:03d}"),
                "plain_feet": words_int(value * 100) + " feet"}

    def altitude(self):
        feet = self.rng.choice(range(1500, 10001, 500))
        return {"value": feet, "spoken": spoken_feet(feet), "plain": words_int(feet) + " feet"}

    def runway(self):
        number = self.rng.randint(1, 36)
        side = self.rng.choice(["", "", "", "L", "R", "C"])
        designator = f"{number:02d}{side}"
        side_word = {"L": " left", "R": " right", "C": " center", "": ""}[side]
        return {"value": designator, "spoken": "runway " + spoken_digits(f"{number:02d}") + side_word,
                "plain": runway_words(designator), "heading_hint": number * 10}

    def pressure(self, dialect):
        if dialect == "faa":
            value = self.rng.randint(2920, 3060)
            return {"unit": "inHg", "value": value / 100,
                    "spoken": "altimeter " + spoken_digits(str(value)),
                    "plain": f"{words_int(value // 100)} point {spoken_digits(f'{value % 100:02d}')} inches of mercury"}
        value = self.rng.randint(975, 1040)
        return {"unit": "hPa", "value": value, "spoken": "qnh " + spoken_digits(str(value)),
                "plain": words_int(value) + " hectopascals"}

    def frequency(self, dialect):
        base = self.rng.randrange(118000, 136976, 25)
        if dialect == "icao_europe" and self.rng.random() < 0.3:
            # 8.33 kHz channel designators end in these pairs; spoken as the channel name.
            base = base // 25 * 25 + self.rng.choice([5, 10, 15])
        text = f"{base // 1000}.{base % 1000:03d}".rstrip("0")
        text = text if "." in text and not text.endswith(".") else text.rstrip(".") + ".0"
        return {"value": text, "spoken": spoken_digits(text, faa=dialect == "faa"),
                "plain": f"{words_int(int(text.split('.')[0]))} point {spoken_digits(text.split('.')[1])} megahertz"}

    def squawk(self, special=False):
        if special:
            code = self.rng.choice(["7700", "7600", "7500"])
        else:
            while True:
                code = "".join(str(self.rng.randint(0, 7)) for _ in range(4))
                if code not in {"7700", "7600", "7500", "7000", "2000", "1200"}:
                    break
        return {"value": code, "spoken": "squawk " + spoken_digits(code), "plain": spoken_digits(code)}

    def wind(self):
        direction = self.rng.randrange(10, 361, 10)
        speed = self.rng.randint(2, 25)
        gust = speed + self.rng.randint(8, 15) if self.rng.random() < 0.15 else None
        spoken = f"wind {spoken_digits(f'{direction:03d}')} degrees {words_int(speed)} knots"
        if gust:
            spoken += f" gusting {words_int(gust)}"
        return {"direction": direction, "speed": speed, "gust": gust, "spoken": spoken}

    def speed(self):
        value = self.rng.randrange(140, 281, 10)
        return {"value": value, "spoken": f"{spoken_digits(str(value))} knots",
                "plain": words_int(value) + " knots"}

    def waypoint(self):
        consonants, vowels = "bdgklmnprstvz", "aeiou"
        name = "".join(
            self.rng.choice(consonants if i % 2 == 0 else vowels) for i in range(5)
        )
        return {"value": name.upper(), "spoken": name}

    def sample(self, dialect):
        """A bundle of candidate values; the generator uses the ones its shape needs."""
        return {
            "dialect": dialect,
            "callsign": self.callsign(dialect),
            "callsign_alt": self.callsign(dialect),
            "heading": self.heading(),
            "flight_level": self.flight_level(),
            "altitude": self.altitude(),
            "runway": self.runway(),
            "pressure": self.pressure(dialect),
            "frequency": self.frequency(dialect),
            "squawk": self.squawk(),
            "wind": self.wind(),
            "speed": self.speed(),
            "waypoint": self.waypoint(),
            "waypoint_alt": self.waypoint(),
            "taxiway": self.rng.choice(LETTERS[:12]) + (" " + DIGITS[self.rng.randint(1, 9)] if self.rng.random() < 0.5 else ""),
            "sequence_number": words_int(self.rng.randint(1, 5)),
        }
