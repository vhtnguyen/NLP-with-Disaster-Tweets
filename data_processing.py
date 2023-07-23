import re
from nltk.corpus import stopwords
import string
import pandas as pd
from bs4 import BeautifulSoup
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pkg_resources
from symspellpy import SymSpell, Verbosity
import requests
import warnings
import numpy as np
from urllib.parse import unquote
import inflect

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

######################


STOP_WORDS = set(stopwords.words('english'))

EMOJIS = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

WORD_TO_KEEP=['url','hashtag','tweetermention','emailaddress','no_hashtag','no_keyword','no_content']

SYM_SPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

DICTINARY_PATH = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
SYM_SPELL.load_dictionary(DICTINARY_PATH, term_index=0, count_index=1)

BIGRAM_PATH = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)
SYM_SPELL.load_bigram_dictionary(BIGRAM_PATH, term_index=0, count_index=2)

URL ="https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json"

BRITSH_TO_AMERICAN_DICT = requests.get(URL).json()

abbreviations = {
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}




def lower_case(text):
    text=text.lower()
    return text.strip()

def remove_html(text):
    soup = BeautifulSoup(text,features="html.parser")
    text = soup.get_text()
    return text.strip()

def expand_contractions(text):
    text=contractions.fix(text)
    return text.strip()

def replace_urls(text):
    pattern = re.compile(r'https?://[^\s/$.?#].[^\s]*')
    text = re.sub(pattern, "url", text)
    return text.strip()

def replace_emails(text):
    pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    text = re.sub(pattern, "emailaddress", text)
    return text.strip()

def replace_mentions(text):
    mention_pattern = r"(?<!\S)@\w+"    
    text = re.sub(mention_pattern, "tweetermention", text)
    return text.strip()

def replace_hashtags(text):
    hashtag_pattern = r"(?<!\S)#(\w+)"    
    text = re.sub(hashtag_pattern, r"\1", text)
    return text.strip()



def repleace_emoji(text):
    for emoji in EMOJIS.keys():
        text = text.replace(emoji, "EMOJI" + EMOJIS[emoji])
    return text.strip()

def handle_unicode(text):
    text = unidecode(text)
    return text.strip()

def handle_special_worlds(text):
    text = re.sub(r"mh370", "missing malaysia airlines flight", text)
    text = re.sub(r"okwx", "oklahoma city weather", text)
    text = re.sub(r"arwx", "arkansas weather", text)    
    text = re.sub(r"gawx", "georgia weather", text)  
    text = re.sub(r"scwx", "south carolina weather", text)  
    text = re.sub(r"cawx", "california weather", text)
    text = re.sub(r"tnwx", "tennessee weather", text)
    text = re.sub(r"azwx", "arizona weather", text)  
    text = re.sub(r"alwx", "alabama Weather", text)
    text = re.sub(r"wordpressdotcom", "wordpress", text)    
    text = re.sub(r"usnwsgov", "united states national weather service", text)
    text = re.sub(r"suruc", "sanliurfa", text)
    

    # Special characters
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏWhen", "When", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"China\x89Ûªs", "China's", text)
    text = re.sub(r"let\x89Ûªs", "let's", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"fromåÊwounds", "from wounds", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)    
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"SuruÌ¤", "Suruc", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"å£3million", "3 million", text)
    text = re.sub(r"åÀ", "", text)

# Contractions
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    
    # ... and ..
    text = text.replace('...', ' ... ')
    if '...' not in text:
        text = text.replace('..', ' ... ')      

    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    return text

def convert_abbrev(text):
    return abbreviations[text.lower()] if text.lower() in abbreviations.keys() else text

def normalize_abbreviations(text):
    matches = re.finditer(r"([A-Z]\.)+", text)
    matched_abbr = [match.group() for match in matches]
    for abbr in matched_abbr:
        text = re.sub(abbr,abbr.replace(".",""), text)
    return text.strip()

def handle_amount_and_percentage(text):
    text = re.sub(r"(₹|\$|£|€|¥)\s?\d+(\.\d+)?", "amountmoney",text)
    text = re.sub(r"\d+(\.\d+)?\s?%", "percentage",text)
    return text.strip()

def handle_numbers(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text.strip()




def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOP_WORDS])

LEMMATIZER = WordNetLemmatizer()

def lemmatize_text(text):
   
    words = [LEMMATIZER.lemmatize(word) for word in text.split()]
    text = ' '.join(words)
    return text.strip()




def correct_spelling_symspell(text):
    words = [
        SYM_SPELL.lookup(
            word, 
            Verbosity.CLOSEST, 
            max_edit_distance=2,
            include_unknown=True
            )[0].term 
        for word in text.split()] 
    text = " ".join(words)
    return text.strip()





def correct_spelling_symspell_compound(text):
    words = [
        word if word in WORD_TO_KEEP else SYM_SPELL.lookup_compound(word, max_edit_distance=2)[0].term
        for word in text.split()
    ] 
    text = " ".join(words)
    return text.strip()



def americanize(text):
    text = [BRITSH_TO_AMERICAN_DICT[word] if word in BRITSH_TO_AMERICAN_DICT else word for word in text.split()]   
    return " ".join(text)


def replace_symbols(text):
    replacements = {
        ">": "greater",
        "<": "less",
        "&": "and",
        "=": "equal",
    }

    for symbol, word in replacements.items():
        text = text.replace(symbol, word)

    return text.strip()

def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text).strip()
    return text

def remove_spaces(text):
    text = text.replace(" ", "")
    return text



def convert_plural_to_singular(text):
    p = inflect.engine()
    words = text.split()
    singular_words = [p.singular_noun(word) or word for word in words]
    return ' '.join(singular_words)

######################
# Các hàm xử lý các feature dạng count
######################

def add_word_count(df):
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

def add_unique_word_count(df):
     df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))

def add_mean_word_length(df):
    df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

def add_chart_count(df):
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))

def add_hashtags_count(df):
    df['hashtag_count'] = df['text'].apply(lambda x: len(re.findall(r"#\w+", x)))

def add_mentions_count(df):
    df['mention_count'] = df['text'].apply(lambda x: len(re.findall(r"(?<!\S)@\w+", x)))

def add_url_count(df):
    df['mention_count'] = df['text'].apply(lambda x: len(re.findall(r"https?://[^\s/$.?#].[^\s]*", x)))


def add_stopword_count(df):
    df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOP_WORDS]))



#################

def process_content(text):
    # chuyển về chữ thường
    text=lower_case(text)

    # thay thế url 
    text=replace_urls(text)

    # decode UTF-8
    text = unquote(text)

    # loại bỏ html
    text=remove_html(text)

    # xóa cái kí hiệu #hashtag -> hashtag
    text=replace_hashtags(text)


    # thay thế email : emailaddress
    text=replace_emails(text)

    # thay thế metion @sasas -> tweetermention
    text=replace_mentions(text)

    # thay thế biểu tưởng cảm xúc :) -> smile
    text=repleace_emoji(text)

    text=convert_abbrev(text)
   # chuyển các từ viết tắt thành dạng đầy đủ vd I'm -> I am
    text=expand_contractions(text)

    # xử lý unicode 
    text=handle_unicode(text)

    # chuyển dạng U.S.A thành USA
    text=normalize_abbreviations(text)

    # xử lý số tiền và số phần trăm $100: amout of money, 15% amont of percent
    text=handle_amount_and_percentage(text)

    # xử lý chữ số 500-> ###, 5 chữ số trở lên-> #####
    text=handle_numbers(text)

    # loại bỏ các stop word
    # text=remove_stopwords(text)

    # chuyển works -> work
    text=lemmatize_text(text)

    # sửa chính tả tách các từ dính nhau thành các từ độc lập
    # xorry -> sorry, wordhand -> word hard
    text=correct_spelling_symspell_compound(text)
    # sửa chính tả
    # text=correct_spelling_symspell(text)

    # chuyển về tiếng Anh-Mỹ
    # colour -> color
    text=americanize(text)

    # thay thế 1 số biểu tượng thành từ vựng
    # & -> and, < less than

    text=replace_symbols(text)

    # xử lý các từ đặc biệt
    text=handle_special_worlds(text)

    # loại bỏ các stop word
    text=remove_stopwords(text)

     # xóa khoảng trắng thừa nếu còn
    text=remove_extra_spaces(text)

    return text

def process_hashtags(text):

    # chuyển về chữ thường
    text=lower_case(text)

    # chuyển các từ viết tắt thành dạng đầy đủ
    text=expand_contractions(text)


    # xử lý số tiền và số phần trăm
    text=handle_amount_and_percentage(text)

    # xử lý chữ số
    text=handle_numbers(text)

    # chuyển works -> work
    text=lemmatize_text(text)

    # sửa chính tả tách các từ dính nhau thành các từ độc lập
    text=correct_spelling_symspell_compound(text)

    # chuyển về tiếng Anh-Mỹ
    text=americanize(text)

    # xử lý các từ đặc biệt
    text=handle_special_worlds(text)

    # loại bỏ các stop word
    text=remove_stopwords(text)

     # xóa khoảng trắng thừa nếu còn
    text=remove_extra_spaces(text)

    return text




#

def add_count_features(df):

    # đếm 
    add_word_count(df)
    add_unique_word_count(df)
    add_mean_word_length(df)
    add_chart_count(df)
    add_hashtags_count(df)
    add_mentions_count(df)
    add_url_count(df)
    add_stopword_count(df)
  
    return df


def handle_missing_values(df):
    df["keyword"]=df["keyword"].fillna("no_keyword")
    df['keyword']= df['keyword'].apply(lower_case)
    

def handle_keyword(text):
 text=lower_case(text)
 text = unquote(text)
 text=remove_html(text)
 text=handle_unicode(text)
 text=handle_special_worlds(text)
 text=convert_plural_to_singular(text)
 text=remove_extra_spaces(text)

 return text



# print(correct_spelling_symspell_compound("xorry"))