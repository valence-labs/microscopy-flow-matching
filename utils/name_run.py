"""A standalone module for generating random, human-readable names."""

import random
import string

# fmt: off

# just a 250 word subset of https://github.com/mrmaxguns/wonderwordsmodule/blob/master/wonderwords/assets/adjectivelist.txt (MIT license)
ADJECTIVES = [
    "quizzical", "highfalutin", "dynamic", "wakeful", "cheerful", "thoughtful", "cooperative", "questionable", "abundant", "uneven",
    "yummy", "juicy", "vacuous", "concerned", "young", "sparkling", "abhorrent", "sweltering", "late", "macho",
    "scrawny", "friendly", "kaput", "divergent", "busy", "charming", "protective", "premium", "puzzled", "waggish",
    "rambunctious", "puffy", "hard", "fat", "sedate", "yellow", "resonant", "dapper", "courageous", "vast",
    "cool", "elated", "wary", "bewildered", "level", "wooden", "ceaseless", "tearful", "cloudy", "gullible",
    "flashy", "trite", "quick", "nondescript", "round", "slow", "spiritual", "brave", "tenuous", "abstracted",
    "colossal", "sloppy", "obsolete", "elegant", "fabulous", "vivacious", "exuberant", "faithful", "helpless", "odd",
    "sordid", "blue", "imported", "ugly", "ruthless", "deeply", "eminent", "reminiscent", "rotten", "sour",
    "volatile", "succinct", "judicious", "abrupt", "learned", "stereotyped", "evanescent", "efficacious", "festive", "loose",
    "torpid", "condemned", "selective", "strong", "momentous", "ordinary", "dry", "great", "ultra", "ahead",
    "broken", "dusty", "piquant", "creepy", "miniature", "periodic", "equable", "unsightly", "narrow", "grieving",
    "whimsical", "fantastic", "kindhearted", "miscreant", "cowardly", "cloistered", "marked", "bloody", "chunky", "undesirable",
    "oval", "nauseating", "aberrant", "stingy", "standing", "distinct", "illegal", "angry", "faint", "rustic",
    "few", "calm", "gorgeous", "mysterious", "tacky", "unadvised", "greasy", "minor", "loving", "melodic",
    "flat", "wretched", "clever", "barbarous", "pretty", "endurable", "handsomely", "unequaled", "acceptable", "symptomatic",
    "hurt", "tested", "long", "warm", "ignorant", "ashamed", "excellent", "known", "adamant", "eatable",
    "verdant", "meek", "unbiased", "rampant", "somber", "cuddly", "harmonious", "salty", "overwrought", "stimulating",
    "beautiful", "crazy", "grouchy", "thirsty", "joyous", "confused", "terrible", "high", "unarmed", "gabby",
    "wet", "sharp", "wonderful", "magenta", "tan", "huge", "productive", "defective", "chilly", "needy",
    "imminent", "flaky", "fortunate", "neighborly", "hot", "husky", "optimal", "gaping", "faulty", "guttural",
    "massive", "watery", "abrasive", "ubiquitous", "aspiring", "impartial", "annoyed", "billowy", "lucky", "panoramic",
    "heartbreaking", "fragile", "purring", "wistful", "burly", "filthy", "psychedelic", "harsh", "disagreeable", "ambiguous",
    "short", "splendid", "crowded", "light", "yielding", "hypnotic", "dispensable", "deserted", "nonchalant", "green",
    "puny", "deafening", "classy", "tall", "typical", "exclusive", "materialistic", "mute", "shaky", "inconclusive",
    "rebellious", "doubtful", "telling", "unsuitable", "woebegone", "cold", "sassy", "arrogant", "perfect", "adhesive",
]

# just a 250 word subset of https://github.com/mrmaxguns/wonderwordsmodule/blob/master/wonderwords/assets/nounlist.txt (MIT license)
NOUNS = [
    "kebab", "presentation", "technologist", "curve", "deadline", "pusher", "emanate", "cost", "zoology", "fritter",
    "heel", "machinery", "shoes", "trust", "potential", "gosling", "expectation", "whack", "boar", "nest",
    "gel", "transaction", "locker", "jury", "kayak", "boyfriend", "residence", "stall", "whirlwind", "sushi",
    "privilege", "bricklaying", "beck", "qualification", "dialogue", "freight", "river", "popularity", "arcade", "apparel",
    "glee", "redhead", "forte", "wholesale", "left", "doctorate", "infix", "handsaw", "people", "intention",
    "nourishment", "campaigning", "temptress", "irrigation", "price", "hour", "dishwasher", "sneeze", "pegboard", "view",
    "choice", "purchase", "parliament", "bull", "currant", "integer", "cohort", "reception", "need", "savory",
    "softdrink", "invite", "college", "zero", "hospitalization", "overnighter", "litigation", "iron", "wren", "overweight",
    "antennae", "team", "loafer", "math", "inspiration", "maybe", "sin", "estrogen", "runway", "bug",
    "mail", "magazine", "conservative", "progression", "dream", "onset", "tuba", "hiring", "atheist", "feed",
    "dash", "modernity", "skill", "spit", "clank", "cylinder", "applause", "eyeball", "muskrat", "analgesia",
    "beard", "croissant", "benefit", "cupcake", "webinar", "invasion", "ignorance", "convention", "currency", "miss",
    "ptarmigan", "sorrel", "conviction", "address", "pannier", "thread", "weekend", "sight", "line", "playwright",
    "difficulty", "account", "paperwork", "personal", "cofactor", "theme", "retailer", "marionberry", "disappointment", "chime",
    "timeline", "connotation", "academy", "aftershock", "prose", "pantology", "revival", "defense", "strobe", "spirituality",
    "secretary", "convertible", "textbook", "activation", "marshmallow", "homonym", "sideburns", "humanity", "advertisement", "bestseller",
    "eleventh", "infant", "macadamia", "flan", "shadowbox", "basketball", "toaster", "stop", "yam", "observatory",
    "cloister", "mallet", "fishbone", "regulator", "liner", "pitching", "aftermath", "spinach", "savings", "validate",
    "headlight", "supplier", "season", "hold", "anyone", "espalier", "bass", "pronunciation", "lobby", "skean",
    "supervisor", "needle", "poppy", "offset", "cravat", "girl", "escort", "expert", "railing", "pen",
    "compassionate", "divine", "swan", "baboon", "immortal", "scow", "apartment", "fatigues", "porcupine", "softening",
    "coleslaw", "sparerib", "cafe", "participant", "grassland", "bath", "director", "grandma", "present", "glen",
    "improvement", "registration", "accessory", "kiss", "coil", "credit", "hydrant", "pupil", "corruption", "seminar",
    "councilman", "beet", "psychology", "executor", "standardization", "dignity", "ownership", "omnivore", "investment", "ale",
    "valley", "grain", "grass", "page", "virus", "pompom", "pledge", "grand", "scrim", "epee",
]

CHARS = string.ascii_lowercase
DIGITS = string.digits

# fmt: on


class NameFormatter(string.Formatter):
    """Custom string formatter to handle random name generation placeholders."""

    def __init__(self, prng: random.Random):
        super().__init__()
        self.prng = prng
        self.placeholder_map = {
            "adjective": ADJECTIVES,
            "noun": NOUNS,
            "char": CHARS,
            "digit": DIGITS,
        }

    def get_value(self, key, args, kwargs):
        del args, kwargs
        if isinstance(key, int):
            raise ValueError(f"Integer placeholders not supported, got: {{{key}}}")

        placeholder_type = key.lower()
        is_uppercase = key.isupper()

        if placeholder_type in self.placeholder_map:
            source_list = self.placeholder_map[placeholder_type]
            value = self.prng.choice(source_list)
            return value.upper() if is_uppercase else value
        else:
            raise ValueError(f"Unknown placeholder: {{{key}}}")

    def format_field(self, value, format_spec):
        del format_spec
        return str(value)


def generate_random_name(
    seed: str | float | int | bytes | bytearray | None = None,
    format_str: str = "{adjective}-{noun}-{char}{char}{digit}{digit}",
) -> str:
    """Generate a random name in the given format.
    If a seed is provided, this function is pure and deterministic.
    Given the same seed and format string, it will produce the same name.

    Args:
        seed: A seed for the random number generator. If None, the current system time is used.
        format_str: A format string for the name.

    Placeholders:
    1. {adjective} - a random adjective (lowercase)
    2. {noun} - a random noun (lowercase)
    3. {char} - a random character a-z (lowercase)
    4. {digit} - a random digit 0-9
    5. Use ALL CAPS for uppercase versions (e.g., {ADJECTIVE}, {NOUN}).

    Escaping:
    - Use double curly braces {{ and }} to include literal braces.

    Examples:
    1. "{adjective}-{noun}-{char}{digit}{digit}" -> "happy-dog-a12"
    2. "{NOUN}{digit}{digit}" -> "FRIEND98"
    3. "#hello%{noun}*{digit}" -> "#hello%book*6"
    4. "Literal {{braces}} with {adjective}" -> "Literal {braces} with quick"

    The default format_str induces a uniform distribution over
    250x250x26x26x10x10 (= ~4.2 billion) possible names.
    This gives an entropy of ~32 bits (assuming a decent prng).

    Due to the birthday paradox, you should probably start worrying about
    name clashes if you're planning on generating tens of thousands of names
    (50% probability of collision at approximately 2^16 = 65k names).
    """
    prng = random.Random(seed)
    formatter = NameFormatter(prng)
    return formatter.format(format_str)


# fmt: off
if __name__ == "__main__":
    """If run as a standalone script, print some examples."""

    print("Some examples:")
    print(f"{generate_random_name()=}")
    print(f"{generate_random_name(seed=None, format_str='{adjective}-{noun}-{char}{digit}{digit}')=}")
    print(f"{generate_random_name(seed=None, format_str='{NOUN}{digit}{digit}')=}")
    print(f"{generate_random_name(seed=None, format_str='{Adjective}_{NOUN}')=}")
    print(f"{generate_random_name(seed=None, format_str='{ADJECTIVE}_{NOUN}')=}")
    print(f"{generate_random_name(seed=None, format_str='Just plain text')=}")
    print(f"{generate_random_name(seed=None, format_str='{char}{char}{digit}{digit}')=}")
    print()
    print("Reproducibility:")
    print(f"{generate_random_name(seed=1)=}")
    print(f"{generate_random_name(seed=1)=}")
    print(f"{generate_random_name(seed='seed string, e.g. hash of some data')=}")
    print(f"{generate_random_name(seed='seed string, e.g. hash of some data')=}")
    print()
    print("You can even use a string representing a path as the seed!")
    print("This is particularly useful if you are using hydra.")
    print(f"{generate_random_name(seed='~/path/to/hydra/output/dir')=}")
