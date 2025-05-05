import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os
import re
from rapidfuzz import process, fuzz
import ast

STANDARD_VOLUME = "liter"
STANDARD_TIME = "minute"
STANDARD_WEIGHT = "gram"
STANDARD_TEMP = "celsius"

# 1. Your tag → cuisine mapping
TAG2CLASS = {
    # North America – United States
    "american": "North America – United States",
    "north-american": "North America – United States",
    "northeastern-united-states": "North America – United States",
    "californian": "North America – United States",
    "native-american": "North America – United States",
    "pennsylvania-dutch": "North America – United States",
    "hawaiian": "North America – United States",

    # North America – Canada
    "canadian": "North America – Canada",
    "british-columbian": "North America – Canada",
    "quebec": "North America – Canada",

    # Central America & Caribbean
    "mexican": "Central America & Caribbean",
    "costa-rican": "Central America & Caribbean",
    "guatemalan": "Central America & Caribbean",
    "caribbean": "Central America & Caribbean",
    "cuban": "Central America & Caribbean",
    "puerto-rican": "Central America & Caribbean",
    "creole": "Central America & Caribbean",

    # South America
    "argentine": "South America",
    "brazilian": "South America",
    "peruvian": "South America",
    "chilean": "South America",
    "colombian": "South America",
    "venezuelan": "South America",
    "ecuadorean": "South America",

    # Europe – Western
    "french": "Europe – Western",
    "english": "Europe – Western",
    "scottish": "Europe – Western",
    "irish": "Europe – Western",
    "welsh": "Europe – Western",
    "dutch": "Europe – Western",
    "belgian": "Europe – Western",
    "austrian": "Europe – Western",
    "german": "Europe – Western",
    "italian": "Europe – Western",
    "portuguese": "Europe – Western",

    # Europe – Northern
    "swedish": "Europe – Northern",
    "norwegian": "Europe – Northern",
    "finnish": "Europe – Northern",
    "icelandic": "Europe – Northern",
    "danish": "Europe – Northern",

    # Europe – Eastern/Central
    "russian": "Europe – Eastern",
    "hungarian": "Europe – Eastern",
    "czech": "Europe – Eastern",
    "georgian": "Europe – Eastern",

    # Middle East & North Africa (MENA)
    "middle-eastern": "Middle East & North Africa",
    "turkish": "Middle East & North Africa",
    "lebanese": "Middle East & North Africa",
    "iranian-persian": "Middle East & North Africa",
    "iraqi": "Middle East & North Africa",
    "palestinian": "Middle East & North Africa",
    "saudi-arabian": "Middle East & North Africa",
    "egyptian": "Middle East & North Africa",
    "moroccan": "Middle East & North Africa",
    "libyan": "Middle East & North Africa",

    # Sub-Saharan Africa
    "south-african": "Sub-Saharan Africa",
    "ethiopian": "Sub-Saharan Africa",
    "nigerian": "Sub-Saharan Africa",
    "angolan": "Sub-Saharan Africa",
    "sudanese": "Sub-Saharan Africa",

    # Asia – East
    "chinese": "Asia – East",
    "beijing": "Asia – East",
    "chinese-new-year": "Asia – East",
    "japanese": "Asia – East",
    "korean": "Asia – East",

    # Asia – Southeast
    "vietnamese": "Asia – Southeast",
    "indonesian": "Asia – Southeast",
    "malaysian": "Asia – Southeast",
    "cambodian": "Asia – Southeast",
    "laotian": "Asia – Southeast",

    # Asia – South
    "pakistani": "Asia – South",
    "nepalese": "Asia – South",

    # Asia – Central & Inner
    "mongolian": "Asia – Central",

    # Oceania & Pacific
    "polynesian": "Oceania & Pacific",

    # Jewish Diaspora
    "jewish-ashkenazi": "Jewish Diaspora",
    "jewish-sephardi": "Jewish Diaspora",

    # Catch-all / generic
    "asian": "Asia – General",
    "european": "Europe – General",
}

# 2. Build a prioritized list of tags, from most specific → most generic
PRIORITIZED_TAGS = [
    # Europe – Western
    "french", "english", "scottish", "irish", "welsh",
    "dutch", "belgian", "austrian", "german", "italian",
    # North America – United States
    "american", "north-american", "northeastern-united-states",
    "californian", "native-american", "pennsylvania-dutch",
    "hawaiian",
    # North America – Canada
    "canadian", "british-columbian", "quebec",
    # Central America & Caribbean
    "mexican", "costa-rican", "guatemalan",
    "caribbean", "cuban", "puerto-rican", "creole",
    # South America
    "argentine", "brazilian", "peruvian",
    "chilean", "colombian", "venezuelan", "ecuadorean",
    # Europe – Northern
    "swedish", "norwegian", "finnish", "icelandic", "danish",
    # Europe – Eastern/Central
    "russian", "hungarian", "czech", "georgian",
    # Europe – Southern
    "portuguese",
    # Middle East & North Africa (MENA)
    "middle-eastern", "turkish", "lebanese",
    "iranian-persian", "iraqi", "palestinian",
    "saudi-arabian", "egyptian", "moroccan", "libyan",
    # Sub-Saharan Africa
    "south-african", "ethiopian", "nigerian",
    "angolan", "sudanese",
    # Asia – East
    "chinese", "beijing", "chinese-new-year",
    "japanese", "korean",
    # Asia – Southeast
    "vietnamese", "indonesian", "malaysian",
    "cambodian", "laotian",
    # Asia – South
    "pakistani", "nepalese",
    # Asia – Central & Inner
    "mongolian",
    # Oceania & Pacific
    "polynesian",
    # Jewish Diaspora
    "jewish-ashkenazi", "jewish-sephardi",
    # Finally, the generic catch-alls
    "asian", "european",
]

# Unit conversion factors to standardized units
unit_conversions = {
    # Volume conversions to liter
    "cup": 0.24,         # 1 cup = 0.24 liters
    "quart": 0.95,       # 1 quart = 0.95 liters
    "mL": 0.001,         # 1 mL = 0.001 liters
    
    # Time conversions to minutes
    "second": 1/60,      # 1 second = 1/60 minutes
    "hours": 60,         # 1 hour = 60 minutes
    "week": 10080,       # 1 week = 10080 minutes
    
    # Weight conversions to grams
    "pound": 453.59,     # 1 pound = 453.59 grams
    "ounce": 28.35,      # 1 ounce = 28.35 grams
    "kg": 1000,          # 1 kg = 1000 grams
    "g": 1,              # 1 g = 1 gram

    # Inch to centimeter conversion
    "inch": 2.54,        # 1 inch = 2.54 cm
    "inches": 2.54,      # 1 inch = 2.54 cm
}

# Mapping of units to their standard category
unit_categories = {
    # Volume units
    "cup": STANDARD_VOLUME,
    "quart": STANDARD_VOLUME,
    "mL": STANDARD_VOLUME,
    "liter": STANDARD_VOLUME,
    
    # Time units
    "second": STANDARD_TIME,
    "minutes": STANDARD_TIME,
    "hours": STANDARD_TIME,
    "week": STANDARD_TIME,
    
    # Weight units
    "pound": STANDARD_WEIGHT,
    "ounce": STANDARD_WEIGHT,
    "kg": STANDARD_WEIGHT,
    "g": STANDARD_WEIGHT,
    "gram": STANDARD_WEIGHT,

    # Measurement units
    "inch": "cm",
    "inches": "cm",
    
    # Temperature units
    "°c": STANDARD_TEMP,
    "°f": STANDARD_TEMP,
    "celsius": STANDARD_TEMP,
    "fahrenheit": STANDARD_TEMP,
}

common_units = list(unit_categories.keys())

typo_corrections = {
    "gram": "g",
    "gm": "g",
    "lb": "pound",
    "oz": "ounce",
    "kilogram": "kg",
    "centimetr": "cm",
    "centimet": "cm",
    "mm": "mL",
    "millimet": "mL",
    "millilit": "mL",
    "centigrad": "°c",
    "litr": "liter",
    "cupof": "cup of",
    "talbespoon": "tablespoon",
    "tablespon": "tablespoon",
    "tablesppoon": "tablespoon",
    "tblpss": "tablespoon",
    "tbso": "tablespoon",
    "tbspn": "tablespoon",
    "tbslp": "tablespoon",
    "tsbp": "tablespoon",
    "tlbsp": "tablespoon",
    "tablestoon": "tablespoon",
    "tablepoon": "tablespoon",
    "teasppon": "teaspoon",
    "teapsoon": "teaspoon",
    "teaspon": "teaspoon",
    "teaspoom": "teaspoon",
    "cu": "cup",
    
    # Temperature corrections
    "f": "°f",
    "fahrenheit": "°f",
    "c": "°c",
    "celsius": "°c",
    "degrees f": "°f",
    "degrees c": "°c",
    "degree f": "°f",
    "degree c": "°c",

    # Times
    "min" : "minutes",
    "minuet": "minutes",
    "minutesthen": "minutes",
    "minutesor": "minutes",
    "minutesyour": "minutes",
    "minuet": "minutes",
    "miniut": "minutes",
    "mimut": "minutes",
    "mionut": "minutes",
    "mintur": "minutes",
    "mkinut": "minutes",
    "mminut": "minutes",
    "munut": "minutes",
    "minuest": "minutes",
    "minunet": "minutes",
    "mintes": "minutes",
    "mutes": "minutes",
    "mutesr": "minutes",
    "minutesr": "minutes",
    "minuteslong": "minutes long",
    "minutesbrush": "minutes brush",
    "minnut": "minutes",
    "minuteuntil": "minutes until",
    "minutesm": "minutes",
    "nminut": "minutes",
    "minit": "minutes",
    "minutu": "minutes",
    "mihnut": "minutes",
    "mintut": "minutes",
    "minutr": "minutes",
    "ninut": "minutes",
    "minutew": "minutes",
    "minutess": "minutes",
    "minutesssssssss": "minutes",
    "minuteswil": "minutes will",
    "seccond": "second",
    "secong": "second",
    "seceond": "second",
    "housr": "hours",
    "houir": "hours",
    "hoursin": "hours",
    "hoursovernight": "hours overnight",
    "secon": "second",
    "seccond": "second",
    "secong": "second",
    "seceond": "second",
    "wk": "week",
    "hr": "hours",
    "b": "lb",
    "z": "oz",
    "″": "inch",
    '"': "inch" 
}

units = ["liter", "minute", "gram", "hours", "cup", "quart", "mL", "second", "week", 
         "pound", "ounce", "kg", "oz", "g", "°c", "°f", "fahrenheit", "degre", "celsius"]

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS_ADDITIONAL = {
    "minutes", "easiest", "ever", "aww", "i", "can", "t", "believe", "it", "s", "stole", "the", "idea", "from","mirj", "andrea", " s ", "andreas",
    "viestad", "andes", "andersen", "an", "ana", "amy", "2 ww points", "on demand", "anelia", "amazing",
    "ashley", "ashton", "amazing", "make", "house", "smell", "malcolm", "amazingly", "killer", "perfect",
    "addictive", "leave", "u", "licking", "ur", "finger", "clean", "th", "recipe", "special", "time", "favorite",
    "aunt", "jane", "soft", "and", "moist", "licking", "famous", "non fruitcake", "true", "later",
    "nonbeliever", "believer", "comfort", "ultimate", "lover", "love", "easy", "ugly", "cc", "uncle", "bill", "tyler",
    "unbelievably", "unbelievable", "healthy", "fat", "free", "un", "melt", "mouth", "ummmmm", "umm", "ummmy", "nummy", "ummmm", "unattended",
    "unbaked", "ultra", "ultimately", "yummy", "rich", "quick", "rachael", "ray", "fail", "party", "florence",
    "fast", "light", "low", "carb", "snack", "wedding", "anniversary", "anne", "marie", "annemarie", "annette", "funicello", "syms",
    "byrn", "mike", "willan", "summer", "autumn", "winter", "spring", "burrel", "anna", "tres", "sweet", "uber",
    "homemade", "ann","best","j", "anite", "anitas", "anman", "angie", "angry", "simple", "difficult", "andy", "andrew", "ancient", "still", "another", "best", "go",
    "grant", "grandma", "amusement", "park", "instruction", "kitchen", "test", "ww", "almost", "empty", "dressing", "instant", "like", "le", "virtually",
    "home", "made", "guilt", "guilty", "delicious", "parfait", "forgotten", "forget", "forevermama", "diet", "can", "real", "former",
    "miss", "fabulous", "forever", "authentic", "fortnum", "mason", "kid", "foolproof", "football", "season", "diabetic",
    "two", "small", "one", "three", "four", "five", "thanksgiving", "dream", "foothill", "paula", "deen", "food", "processor", "safari", "processor",
    "traditional", "forbidden", "flavorful", "grandmag", "grandmama", "grandmaman", "grandma", "grandmom", "lena", "alicia", "alisa", "alice", "ali", "bit", "different",
    "eat", "family", "global", "gourmet", "yam", "yam", "emotional", "balance", "tonight", "feel", "cooking", "got", "birthday", "air", "way", "mr", "never", "weep", "half",
    "anything", "pour", "put", "fork", "say", "stove", "top", "thought", "prize", "winning", "add", "ad", "good", "better", "da", "style", "even", "bran", "fake", "fire", "beautiful"
    "l", "game", "day", "hate", "world", "minute", "type", "starbucks", "biggest", "dressed", "summertime", "elmer", "johnny", "depp", "c", "p", "h", "clove", "er", "star", "week",
    "affair", "elegant", "student", "z", "whole", "lotta", "w", "z", "b", "aaron", "craze", "a", "abc", "absolute", "absolut", "absolutely", "perfection", "delightful", "lazy", "morning",
    "abuelo", "abuelito", "abuelita", "abuela", "acadia", "accidental", "adam", "little", "interest", "addicting", "addie", "adele", "adelaide", "adi", "adie", "adriana",
    "adult", "affordable", "alison", "holst", "purpose", "allegheny", "allegedly", "original", "allergic", "ex", "allergy", "allergen", "allen", "poorman", "backyard",
    "alton", "brown", "whatever", "anthony", "anytime", "april", "fool", "ya", "fooled", "sandra", "lee", "edna", "emma", "emy", "evy", "eva", 'evelyn', "fannie", "fanny", "flo", "gladys", "helen", "grace", "ira", "irma",
    "isse", "jean", "janet", "jenny", "juju", "judy", "kathy", "kathi", "kellie", "kelly", "laura", "lee", "kay", "kathleen", "laura", "lee", "lesley", "lil", "linda", "liz", "lois", "louisse",
    "mag", 'martguerite', "margie", "marge", "maggie", "martha", "marylin", "marion", "mary", "marthy", "melody", "michel", "meda", "millie", "muriel", "myrna", "nelda", "nancy", "paulie", "phillis", "rae", "rebecca",
    "rose", "sadie", "sarah", "sara", "sue", "susan", "teresa", "theresa", "auntie", "em", "barbara", "barb", "irene", "lolo", "lori", "lu", "maebelle",
    "aunty", "aussie", "aurora", "austin", "l", "q"
    
    }
STOP_WORDS.update(STOP_WORDS_ADDITIONAL) 

def fahrenheit_to_celsius(f: float):
    return round((f - 32) * 5.0 / 9.0, 2)

class BasePreprocessing:
    def __init__(self):
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def load(self, filename: str) -> pd.DataFrame:
        data  = pd.read_csv('../../data/RAW_recipes.csv')
        data.set_index('id', inplace=True)
        columns = ["tags", "steps", "ingredients", "nutrition"]
        for i in columns:
            data[i] = data[i].apply(ast.literal_eval)
        data.drop(columns=["contributor_id", "submitted"], inplace=True, errors="ignore")
        data.dropna(subset=["name"], inplace=True)
        data = data[data['minutes'] < 300]
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Cleaning recipe names...")
        data['name'] = self._clean_recipe_names(data['name'])

        data.dropna(subset=['name', 'description'], inplace=True)
        data.reset_index(inplace=True)

        print("Cleaning description, steps, tags & ingredients")
        data["description"] = data["description"].apply(lambda x: x.lower())

        for c in ["tags", "steps", "ingredients"]:
            data[c] = data[c].apply(lambda x : [s.lower() for s in x])

        data["steps_strings"] = data["steps"].apply(lambda x : ' '.join(x))

        print("Standardize units...")
        data["steps_string_standardize"] = data["steps_strings"].apply(self._standardize_units)

        units = ["liter", "minute", "gram", "hours", "cup", "quart", "mL", "second", "week", 
                 "pound", "ounce", "kg", "oz", "g", "°c", "°f", "fahrenheit", "degre", "celsius"]

        print("Apply number unit diff...")
        diffs = data.apply(self._number_unit_diff, axis=1, result_type='expand')

        print("Concatenate results...")
        result = pd.concat([data, diffs], axis=1)

        data["ingredients_text"] = data["ingredients"].apply(lambda x: ' '.join(x))
        data["ingredients_text"] = data["ingredients"].astype(str)
        
        data["tags_text"] = data["tags"].apply(lambda x: ' '.join(x))
        data["tags_text"] = data["tags"].astype(str)

        print("Finding cuisine for each recipe...")
        _tag_rank = { tag: idx for idx, tag in enumerate(PRIORITIZED_TAGS) }

        data['cuisine'] = data['tags'].apply(self._map_tags_to_cuisine)
        data.dropna(subset=['cuisine'], inplace=True)

        print("Expand Nutrition data...")
        data = self._expand_nutrition_column(data)
        data.drop(columns=['nutrition_score'], inplace=True, errors='ignore')
        data.drop(columns=['ingredients', 'steps', 'steps_strings', 'tags'], inplace=True, errors='ignore')

        return data
    
    def _clean_recipe_names(self, recipes):
        cleaned_recipes = []
        
        for recipe in recipes:
    
            recipe = recipe.lower()
            recipe = re.sub(r'[^a-z\s]', '', recipe)
            
            recipe_words = recipe.split()
            
            # Lemmatize first
            recipe_words = [self.lemmatizer.lemmatize(word) for word in recipe_words]

            # Then remove stopwords
            recipe_words = [word for word in recipe_words if word not in STOP_WORDS]
            
            cleaned_recipe = " ".join(recipe_words)
            cleaned_recipes.append(cleaned_recipe)
        
        return cleaned_recipes

    def _standardize_temperature(self, text: str):
        """
        Process temperature mentions in text, converting Fahrenheit to Celsius,
        but preserving temperatures that are already in Celsius.
        Also avoids confusing time durations with temperatures.
        """
        # First handle explicit Celsius (just standardize format without changing the value)
        # Handle this BEFORE Fahrenheit to avoid double conversion
        c_pattern = r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*c|°c|\^c|celsius)'
        result_text = re.sub(c_pattern, 
                         lambda m: f"{float(m.group(1))} {STANDARD_TEMP}", 
                         text, 
                         flags=re.IGNORECASE)
        
        # Also handle direct C notation
        direct_c_pattern = r'(\d+(?:\.\d+)?)([°]?c\b)'
        result_text = re.sub(direct_c_pattern, 
                         lambda m: f"{float(m.group(1))} {STANDARD_TEMP}", 
                         result_text, 
                         flags=re.IGNORECASE)
        
        # Now handle explicit Fahrenheit temperatures, which we definitely want to convert
        # Handle explicit Fahrenheit with degree symbol or text
        f_pattern = r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*f|°f|\^f|fahrenheit)'
        
        def process_temp(match):
            temp_value = float(match.group(1))
            return f"{fahrenheit_to_celsius(temp_value)} {STANDARD_TEMP}"
        
        result_text = re.sub(f_pattern, process_temp, result_text, flags=re.IGNORECASE)
        
        # Handle direct F notation
        direct_f_pattern = r'(\d+(?:\.\d+)?)([°]?f\b)'
        result_text = re.sub(direct_f_pattern, process_temp, result_text, flags=re.IGNORECASE)
        
        # For standalone "350 degrees" without explicit F/C, we should be careful
        # We'll only convert if we can strongly infer it's Fahrenheit (like high cooking temps)
        pattern_standalone_degrees = r'(\d+(?:\.\d+)?)\s*(?:degrees?|°)(?!\s*[fc]|\s*fahrenheit|\s*celsius)'
        
        def process_ambiguous_temp(match):
            value = float(match.group(1))
            # Only convert if very likely Fahrenheit (high cooking temps > 200)
            if value > 200:
                return f"{fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
            # For temperatures that could be either F or C (e.g., 100 degrees),
            # preserve the original text to avoid incorrect conversions
            return match.group(0)
        
        result_text = re.sub(pattern_standalone_degrees, process_ambiguous_temp, result_text, flags=re.IGNORECASE)
        
        # Then handle cooking context temperatures (preheat, heat, bake, etc.) CAREFULLY
        # We need to avoid matching phrases like "bake 15 minutes"
        # This pattern specifically looks for temperatures WITHOUT explicit Celsius indication
        cooking_temp_pattern = r'((?:preheat|heat|oven|temperature|temp)(?:\s+to)?)\s+(\d+(?:\.\d+)?)(?:\s*(?:degrees?|°)|\b)(?!\s*(?:minute|min|hour|sec|day|week))(?!\s*(?:c|celsius|°c))'
        
        def process_cooking_temp(match):
            context = match.group(1)
            value = float(match.group(2))
            # For cooking temperatures:
            # - Values below 100: Could be C, don't convert
            # - Values 100-200: Ambiguous zone, examine more carefully
            # - Values above 200: Very likely F, convert to C
            if value > 200:
                return f"{context} {fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
            # For ambiguous or likely Celsius values, preserve original
            return match.group(0)
        
        result_text = re.sub(cooking_temp_pattern, process_cooking_temp, result_text, flags=re.IGNORECASE)
        
        # Special case for "bake at X" or "cook at X" where X is a temperature
        # Only match cases NOT explicitly marked as Celsius
        bake_at_pattern = r'((?:bake|cook)(?:\s+at)?)\s+(\d+(?:\.\d+)?)(?:\s*(?:degrees?|°)|\b)(?!\s*(?:minute|min|hour|sec|day|week))(?!\s*(?:c|celsius|°c))'
        
        def process_bake_temp(match):
            context = match.group(1)
            value = float(match.group(2))
            # Similar logic as cooking temperatures
            # Only convert values that are very likely to be Fahrenheit
            if value > 200:
                return f"{context} {fahrenheit_to_celsius(value)} {STANDARD_TEMP}"
            # For ambiguous temperatures or temperatures likely in Celsius already, preserve original
            return match.group(0)
        
        result_text = re.sub(bake_at_pattern, process_bake_temp, result_text, flags=re.IGNORECASE)
        
        return result_text

    def _standardize_measurements(self, text: str) -> str:
        """
        Handle measurement-specific standardizations, especially for dimensions like 9x5"
        """
        # Keep the dimension format (NxM) but convert each number from inches to cm
        dimension_pattern = r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:"|″|inch(?:es)?)?'
        result = re.sub(dimension_pattern, 
                        lambda m: f"{float(m.group(1)) * 2.54}x{float(m.group(2)) * 2.54} cm", 
                        text)

        # Handle single inch measurements
        inch_pattern = r'(\d+(?:\.\d+)?)(?:"|″|inch(?:es)?)'
        result = re.sub(inch_pattern, 
                        lambda m: f"{float(m.group(1)) * 2.54} cm", 
                        result)

        return result
    
    def _correct_term(self, word: str):
        """Apply fuzzy matching to correct typos in unit terms"""
        # If it is a number return the word
        if not any(c.isalpha() for c in word):
            return word
    
        # Check if in mapping
        if word in typo_corrections:
            return typo_corrections[word]
          
        # Fuzzy matching
        match, score, _ = process.extractOne(word, common_units, scorer=fuzz.ratio)
        if score > 80:
            return match
        return word

    def _parse_range(self, word: str, next_word=None):
        """
        Detects numeric ranges like "2-3" or "2 to 3" and returns their mean as a float.
        E.g. => 2-3 kgs becomes 2.5 kgs
        """
        if re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", word):  # "2-3"
            start, end = map(float, word.split("-"))
            return (start + end) / 2
        if next_word and word.isdigit() and next_word == "to":
            return "to" 
        return None

    def _standardize_units(self, text: str) -> str:
        """
        Main function to standardize all units in text.
        This refactored approach processes different unit types in separate passes.
        """
        # Step 1: First convert temperatures (to avoid conflicts with other patterns)
        result = self._standardize_temperature(text)
        
        # Step 2: Handle measurement standardizations (inches, dimensions)
        result = self._standardize_measurements(result)

        # Step 3: Now process the remaining units
        words = result.lower().split()
        result_words = []
        i = 0
        
        while i < len(words):
            word = words[i]
            next_word = words[i+1] if i + 1 < len(words) else ""
            next2_word = words[i+2] if i + 2 < len(words) else ""
            next3_word = words[i+3] if i + 3 < len(words) else ""
    
            # Handle fractions like "1 / 2 inch"
            if (
                i + 2 < len(words)
                and re.match(r"^\d+(\.\d+)?$", word)
                and words[i+1] == "/"
                and re.match(r"^\d+(\.\d+)?$", words[i+2])
            ):
                numerator = float(word)
                denominator = float(words[i+2])
                fraction_value = numerator / denominator
                
                # Check if there's a unit after the fraction
                if i + 3 < len(words):
                    corrected_unit = self._correct_term(words[i+3])
                    if corrected_unit in unit_categories:
                        category = unit_categories[corrected_unit]
                        converted = fraction_value * unit_conversions.get(corrected_unit, 1)
                        result_words.append(f"{converted} {category}")
                        i += 4  # Move past the fraction and the unit
                        continue
                
                # If no unit or unrecognized unit, just keep the fraction as a decimal
                result_words.append(str(fraction_value))
                i += 3
                continue
    
            # Handle "2-3 kg"
            value = self._parse_range(word, next_word)
            if isinstance(value, float) and next_word:
                corrected_unit = self._correct_term(next_word)
                if corrected_unit in unit_categories:
                    category = unit_categories[corrected_unit]
                    converted = value * unit_conversions.get(corrected_unit, 1)
                    result_words.append(f"{converted} {category}")
                    i += 2
                    continue
    
            # Handle "2 to 3 kg"
            if value == "to" and next2_word.replace('.', '', 1).isdigit() and next3_word:
                average = (float(word) + float(next2_word)) / 2
                corrected_unit = self._correct_term(next3_word)
                if corrected_unit in unit_categories:
                    category = unit_categories[corrected_unit]
                    converted = average * unit_conversions.get(corrected_unit, 1)
                    result_words.append(f"{converted} {category}")
                    i += 4
                    continue
    
            # Handle regular value + unit
            if re.match(r"^\d+(\.\d+)?$", word) and next_word:
                corrected_unit = self._correct_term(next_word)
                if corrected_unit in unit_categories:
                    category = unit_categories[corrected_unit]
                    converted = float(word) * unit_conversions.get(corrected_unit, 1)
                    result_words.append(f"{converted} {category}")
                    i += 2
                    continue
    
            # Default: stem and append
            result_words.append(self.ps.stem(self._correct_term(word)))
            i += 1
    
        return " ".join(result_words)
    
    
    # Function to find differences only on number + unit
    def _number_unit_diff(self, row):
        pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(units) + r')'
    
        matches1 = set(re.findall(pattern, row['steps_string_standardize']))
        matches2 = set(re.findall(pattern, row['steps_strings']))
    
        diff1 = matches1 - matches2
        diff2 = matches2 - matches1
        
        return {
            'only_in_standardize': list(diff1),
            'only_in_not_standardize': list(diff2)
        }

    def _map_tags_to_cuisine(self, tags: list):
        known = [t for t in tags if t in TAG2CLASS]
        if not known:
            return None
        best = min(known, key=lambda t: _tag_rank.get(t, float('inf')))
        return TAG2CLASS[best]

    def _expand_nutrition_column(self, data: pd.DataFrame) -> pd.DataFrame:
        data['nutrition'] = data['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        if data['nutrition'].apply(lambda x: isinstance(x, list)).all():
            data[['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']] = pd.DataFrame(data['nutrition'].to_list(), index=data.index)

            data.drop(columns=['nutrition'], inplace=True)
            
        return data
