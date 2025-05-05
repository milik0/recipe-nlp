import pandas as pd
import numpy as np
import nltk
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
import os

nltk.download('stopwords')
nltk.download('wordnet')

TAG2CLASS = {
    # North America – United States
    "american": ["North America – United States"],
    "north-american": ["North America – United States"],
    "northeastern-united-states": ["North America – United States"],
    "californian": ["North America – United States"],
    "native-american": ["North America – United States"],
    "pennsylvania-dutch": ["North America – United States"],
    "hawaiian": ["North America – United States"],

    # North America – Canada
    "canadian": ["North America – Canada"],
    "british-columbian": ["North America – Canada"],
    "quebec": ["North America – Canada"],

    # Central America & Caribbean
    "mexican": ["Mexican", "Central America & Caribbean"],
    "costa-rican": ["Central America & Caribbean"],
    "guatemalan": ["Central America & Caribbean"],
    "caribbean": ["Central America & Caribbean"],
    "cuban": ["Central America & Caribbean"],
    "puerto-rican": ["Central America & Caribbean"],
    "creole": ["Central America & Caribbean"],

    # South America
    "argentine": ["South America"],
    "brazilian": ["South America"],
    "peruvian": ["South America"],
    "chilean": ["South America"],
    "colombian": ["South America"],
    "venezuelan": ["South America"],
    "ecuadorean": ["South America"],

    # Europe – Western
    "french": ["French", "Europe – Western"],
    "english": ["Europe – Western"],
    "scottish": ["Europe – Western"],
    "irish": ["Europe – Western"],
    "welsh": ["Europe – Western"],
    "dutch": ["Europe – Western"],
    "belgian": ["Europe – Western"],
    "austrian": ["Europe – Western"],
    "german": ["Europe – Western"],
    "italian": ["Italian", "Europe – Western"],
    "portuguese": ["Europe – Western"],
    "spanish": ["Spanish", "Europe – Western"],
    "greek": ["Greek", "Europe – Western"],

    # Europe – Northern
    "swedish": ["Europe – Northern"],
    "norwegian": ["Europe – Northern"],
    "finnish": ["Europe – Northern"],
    "icelandic": ["Europe – Northern"],
    "danish": ["Europe – Northern"],

    # Europe – Eastern/Central
    "russian": ["Europe – Eastern"],
    "hungarian": ["Europe – Eastern"],
    "czech": ["Europe – Eastern"],
    "georgian": ["Europe – Eastern"],

    # Middle East & North Africa (MENA)
    "middle-eastern": ["Middle East & North Africa"],
    "turkish": ["Middle East & North Africa"],
    "lebanese": ["Middle East & North Africa"],
    "iranian-persian": ["Middle East & North Africa"],
    "iraqi": ["Middle East & North Africa"],
    "palestinian": ["Middle East & North Africa"],
    "saudi-arabian": ["Middle East & North Africa"],
    "egyptian": ["Middle East & North Africa"],
    "moroccan": ["Middle East & North Africa"],
    "libyan": ["Middle East & North Africa"],
    "algerian": ["Middle East & North Africa"],
    "tunisian": ["Middle East & North Africa"],

    # Asia – East
    "chinese": ["Chinese", "Asia – East"],
    "beijing": ["Chinese", "Asia – East"],
    "chinese-new-year": ["Chinese", "Asia – East"],
    "japanese": ["Japanese", "Asia – East"],
    "korean": ["Korean", "Asia – East"],
    "mongolian": ["Asia – East"],

    # Asia – Southeast
    "vietnamese": ["Asia – Southeast"],
    "indonesian": ["Asia – Southeast"],
    "malaysian": ["Asia – Southeast"],
    "cambodian": ["Asia – Southeast"],
    "laotian": ["Asia – Southeast"],
    "thai": ["Thai", "Asia – Southeast"],
    "polynesian": ["Asia – Southeast"],

    # Asia – South
    "pakistani": ["Asia – South"],
    "nepalese": ["Asia – South"],
    "indian": ["Indian", "Asia – South"],

    # Jewish Diaspora
    "jewish-ashkenazi": ["Jewish Diaspora", "Middle East & North Africa"],
    "jewish-sephardi": ["Jewish Diaspora", "Middle East & North Africa"],

    # Catch-all / generic
    "asian": ["Asia – General"],
    "european": ["Europe – General"],
}
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
    "japanese", "korean", "mongolian",
    # Asia – Southeast
    "vietnamese", "indonesian", "malaysian",
    "cambodian", "laotian",
    "polynesian",
    # Asia – South
    "pakistani", "nepalese",
    # Jewish Diaspora
    "jewish-ashkenazi", "jewish-sephardi",
    # Finally, the generic catch-alls
    "asian", "european",
]

ADD_STOPWORDS = {
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

class TFIDFPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(ADD_STOPWORDS) 
        self.steps_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.ingredients_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.min_samples = 50
        self.valid_cuisines = None
        self._tag_rank = {tag: idx for idx, tag in enumerate(PRIORITIZED_TAGS)}

    def load(self, filepath: str) -> pd.DataFrame:
        """Load the recipe dataset from a CSV file."""
        print("Loading dataset...")
        data = pd.read_csv(filepath)
        data.set_index('id', inplace=True)
        columns = ["tags", "steps", "ingredients", "nutrition"]

        for col in columns:
            data[col] = data[col].apply(ast.literal_eval)

        data.drop(columns=["contributor_id", "submitted"], inplace=True, errors="ignore")
        data.dropna(subset=["name"], inplace=True)
        data = data[data['minutes'] < 300]

        return data

    def _clean_recipe_names(self, recipes: List[str]):
        cleaned_recipes = []
        
        for recipe in recipes:
            recipe = recipe.lower()
            recipe = re.sub(r'[^a-z\s]', '', recipe)
            recipe_words = recipe.split()
            recipe_words = [self.lemmatizer.lemmatize(word) for word in recipe_words]            
            recipe_words = [word for word in recipe_words if word not in self.stop_words]
            cleaned_recipe = " ".join(recipe_words)
            cleaned_recipes.append(cleaned_recipe)
        
        return cleaned_recipes
        
    def _map_tags_to_cuisines(self, tags: List[str], cleaned_name):
        known = set()

        for t in tags:
            if t in TAG2CLASS:
                known.add(t)
    
        for tag in TAG2CLASS.keys():
            if tag.lower() in cleaned_name.lower():
                known.add(tag)
    
        if not known:
            return []
        
        all_cuisines = []
        for tag in known:
            all_cuisines.extend(TAG2CLASS[tag])
    
        unique_cuisines = []
        for cuisine in all_cuisines:
            if cuisine not in unique_cuisines:
                unique_cuisines.append(cuisine)
    
        def get_cuisine_priority(cuisine):
            cuisine_tags = []
            for tag in known:
                if cuisine in TAG2CLASS[tag]:
                    cuisine_tags.append(tag)
            
            best_rank = min(self._tag_rank.get(tag, float('inf')) for tag in cuisine_tags)
            return best_rank
    
        sorted_cuisines = sorted(unique_cuisines, key=get_cuisine_priority)
        
        return sorted_cuisines

    def _common_preprocessing(self, data: pd.DataFrame):
        print("Cleaning Recipe name...")
        data['name'] = self._clean_recipe_names(data['name'])

        data.dropna(subset=['name', 'description'], inplace=True)
        data.reset_index(inplace=True)

        print("Cleaning description, tags, steps, ingredients...")
        data["description"] = data["description"].apply(lambda x: x.lower())
        columns = ["tags", "steps", "ingredients"]
        for c in columns:
            data[c] = data[c].apply(lambda x: [s.lower() for s in x])
        
        data["steps_strings"] = data["steps"].apply(lambda x: ' '.join(x))
        data["ingredients_text"] = data["ingredients"].apply(lambda x: ' '.join(x))
        data["tags_text"] = data["tags"].apply(lambda x: ' '.join(x))
        
        print("Map tag to cuisine to build target...")
        data['cuisines'] = data.apply(lambda row: self._map_tags_to_cuisines(row['tags'], row['name']), axis=1)

        # Filter recipes with no cuisines
        data = data[data['cuisines'].apply(len) > 0]
        
        return data

    def fit(self, data: pd.DataFrame):
        """Fit preprocessing components on the training data."""
        print("Starting fit process...")
        data = self._common_preprocessing(data)
        
        print("Determine valid cuisines based on minimum sample count")
        cuisine_counts = data['cuisines'].value_counts()
        self.valid_cuisines = cuisine_counts[cuisine_counts >= self.min_samples].index.tolist()
        print(f"Identified {len(self.valid_cuisines)} valid cuisines with at least {self.min_samples} samples each.")
        
        print("Filter data to only include valid cuisines")
        data = data[data['cuisines'].isin(self.valid_cuisines)].copy()
        
        print("Extract primary cuisine")
        data['primary_cuisine'] = data['cuisines'].apply(lambda cuisines: cuisines[0] if len(cuisines) > 0 else None)
        
        print("Fitting TF-IDF vectorizers...")
        self.steps_vectorizer.fit(data["steps_strings"])
        self.ingredients_vectorizer.fit(data["ingredients_text"])
        
        print("Fitting label encoder...")
        self.label_encoder.fit(data['primary_cuisine'])
        
        return self

    def transform(self, data):
        """Transform data using the fitted preprocessing components."""
        print("Starting transform process...")
        data = self._common_preprocessing(data)
        
        # Filter data to only include valid cuisines
        if self.valid_cuisines is None:
            raise ValueError("The preprocessor has not been fitted. Call fit() first.")
        
        data = data[data['cuisines'].isin(self.valid_cuisines)].copy()
        
        # Extract primary cuisine
        data['primary_cuisine'] = data['cuisines'].apply(lambda cuisines: cuisines[0] if len(cuisines) > 0 else None)
        
        # Transform text data using fitted vectorizers
        print("Applying TF-IDF transformations...")
        steps_features = self.steps_vectorizer.transform(data["steps_strings"]).toarray()
        ingredients_features = self.ingredients_vectorizer.transform(data["ingredients_text"]).toarray()
        
        # Extract numerical features
        numerical_features = data[['n_steps', 'n_ingredients']].values
        
        # Combine all features
        print("Building feature matrix...")
        feature_matrices = [
            steps_features,
            ingredients_features,
            numerical_features
        ]

        X = np.hstack(feature_matrices)
        y = self.label_encoder.transform(data['primary_cuisine'])
        return X, y

    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(data)
        return self.transform(data)
    
    def save_pipeline(self, directory="./preprocessor"):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "steps_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.steps_vectorizer, f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.ingredients_vectorizer, f)

        with open(os.path.join(directory, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "wb") as f:
            pickle.dump(self.valid_cuisines, f)

        print(f"Preprocessor saved to {directory}")
        
    def load_pipeline(self, directory="./preprocessor"):
        with open(os.path.join(directory, "steps_vectorizer.pkl"), "rb") as f:
            self.steps_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "rb") as f:
            self.ingredients_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "rb") as f:
            self.valid_cuisines = pickle.load(f)

        print(f"Preprocessor loaded from {directory}")
        return self
