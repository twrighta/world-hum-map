import pandas as pd
import numpy as np
import warnings
import re
from keybert import KeyBERT

#kw_model = KeyBERT()  # Useful for extracting key words / phrases


warnings.simplefilter("ignore")
# Read and check dataframe
path = "C:/Users/tomwr/Datascience/Datasets/Tabular/World Hum Map/hummap_unprocessed.csv"
df = pd.read_csv(path,
                 parse_dates=["full_date"])

# df = df.head(100)  # For testing.
NUMERIC_COLS = list(df.select_dtypes("number").columns)
STRING_COLS = list(df.select_dtypes("string").columns)


for column in STRING_COLS:
    df[column] = df[column].astype("string")
    df[column] = df[column].replace({"None": "",
                                     np.nan: ""})

# qualifications
df["qualifications"] = df["qualifications"].astype("string")
df["qualifications"] = df["qualifications"].replace({None: "",
                                                     np.nan: ""})
df["medication"] = df["medication"].astype("string")
df["medication"] = df["medication"].replace({None: "",
                                             np.nan: ""})

df["sound_desc"] = df["sound_desc"].astype("string")
df["sound_desc"] = df["sound_desc"].replace({None: "",
                                             np.nan: ""})

df["antibiotic_use"] = df["antibiotic_use"].astype("string")
df["antibiotic_use"] = df["antibiotic_use"].replace({None: "",
                                                     np.nan: ""})

df["soft_drink_consumption"] = df["soft_drink_consumption"].astype("string")
df["soft_drink_consumption"] = df["soft_drink_consumption"].replace({None: "",
                                                                     np.nan: ""})

df["change_with_weather_flag"] = df["change_with_weather_flag"].astype("string")
df["change_with_weather_flag"] = df["change_with_weather_flag"].replace({None: "",
                                                                         np.nan: ""})
df["change_with_season_flag"] = df["change_with_season_flag"].astype("string")
df["change_with_season_flag"] = df["change_with_season_flag"].replace({None: "",
                                                                       np.nan: ""})

df["high_altitude"] = df["high_altitude"].astype("string")
df["high_altitude"] = df["high_altitude"].replace({None: "",
                                                   np.nan: ""})

df["heard_elsewhere_flag"] = df["heard_elsewhere_flag"].astype("string")
df["heard_elsewhere_flag"] = df["heard_elsewhere_flag"].replace({None: "",
                                                   np.nan: ""})

df["prev_loud_work_flag"] = df["prev_loud_work_flag"].astype("string")
df["prev_loud_work_flag"] = df["prev_loud_work_flag"].replace({None: "",
                                                   np.nan: ""})

df["pulse_flag"] = df["pulse_flag"].astype("string")
df["pulse_flag"] = df["pulse_flag"].replace({None: "",
                                                   np.nan: ""})

df["ever_stop"] = df["ever_stop"].astype("string")
df["ever_stop"] = df["ever_stop"].replace({None: "",
                                                   np.nan: ""})
## Work through each object column and fix ######################################################

# Gender
binary_genders = ["Male", "Female"]
df["gender"] = df["gender"].apply(lambda gender: gender if gender in binary_genders else "Non Binary")  # Set to non binary if not male or female
# print(df["gender"].value_counts())



# sound_desc - keeping sound_desc as full description, also creating a few extra columns based on common description words/topics
low_pitch_list = ["low", "low pitch", "low-frequency", "rumble", "deep", "hum", "idling", "idle", "drone"]
high_pitch_list = ["high pitch", "high", "whining", "pierc"]
quiet_list = ["faint", "quiet", "difficult to hear"]
loud_list = ["loud", "noise", "noisy", "deafening"]
def pitch_heard(sound_desc):
    sound = sound_desc.lower()
    try:
        if any(substring.lower() in sound for substring in low_pitch_list):
            return "Low"
        elif any(substring.lower() in sound for substring in high_pitch_list):
            return "High"
        else:
            return "Not described"
    except:
        return "Not described"

df["pitch"] = df["sound_desc"].apply(pitch_heard)
df = df.rename(columns={"sound_desc": "sound_desc_full"})

def volume_heard(sound_desc):
    sound = sound_desc.lower()
    try:
        if any(substring.lower() in sound for substring in quiet_list):
            return "Quiet"
        elif any(substring.lower() in sound for substring in loud_list):
            return "Loud"
        else:
            return "Not described"
    except:
        return "Not described"

df["volume"] = df["sound_desc_full"].apply(volume_heard)


# effort - replacing for more consistency
df.replace({"effort": {"Very little": "Minimal",
                       "Some effort": "Low",
                       "Quite a bit of effort": "Moderate",
                       "A lot of effort": "High"}},
           inplace=True)
df["effort"].replace(np.nan,
                     "Minimal effort",
                     inplace=True)

for column in STRING_COLS:
    df[column] = df[column].astype("string")
    df[column] = df[column].replace({"None": "",
                                     np.nan: ""})

# qualification Substrings to try match on
doctorate_substrings = ["professor", "doctor", "dr", "phD"]
graduate_substrings = ["MS", "BS", "BSc", "BA", "MA", "M.D", "M.A.", "b.a", "b,s" "MRes", "MPhil", "degree", "Bachelors", "Masters", "graduate",
                       "university", "research", "MFA", "math", "chemistry", "biology", "physics", "computer science", "law", "batchelor"]
non_skilled_substrings = ["no", "none", "non", "don\'t", "N/A", "n/a", "nope", "Keine", "No scientific training", "untrained", "nothing", "nill", "?",
                          "Zilch"]
professional_substrings = ["technician", "mechanic", "trade", "engineer", "military", "IT", "Electrician", "Law",
                           "Nurse", "chartered", "analyst", "Veterinarian", "Technician", "registered",
                           "Professional", "certified", "qualified",  "civil servant", "teacher", "lawyer", "developer",
                           "police", "nurse", "analyst", ]


# Outside of these, professional
def categorize_qualification(qualification):
    qualification = qualification.lower()
    try:
        if any(substring.lower() in qualification for substring in doctorate_substrings):
            return "Doctorate"
        elif any(substring.lower() in qualification for substring in graduate_substrings):
            return "Graduate"
        elif any(substring.lower() in qualification for substring in non_skilled_substrings):
            return "Non-skilled"
        elif any(substring.lower() in qualification for substring in professional_substrings):
            return "Professional"
        else:
            return "Unknown"
    except:
        return "Unknown"


df["qualification_type"] = df["qualifications"].apply(categorize_qualification)
df.drop(columns=["qualifications"],
        inplace=True)


# first_heard - just remove any non-numeric text
def get_year_first_heard(first_heard):
    year_pattern = r"[0-9]{4}"
    try:
        search = re.search(year_pattern, first_heard).group(0)
        return search
    except AttributeError:
        return "Unknown"


# still_hearing
currently_hearing = ["yes", "always", "night and day", "day and night"]  # Always hearing
intermittent_hearing = ["sometimes", "now and again", "off and on", "bouts", "occasionally", "time to time",
                        "comes and goes", "intermittent", "not all the time", "regularly"]
rare_or_never_hearing = ["never", "no more", "no longer", "not anymore", "rarely", "moved away"]


def categorize_still_hearing(still_hearing):
    still_hearing = still_hearing.lower()
    try:
        if any(substring.lower() in still_hearing for substring in currently_hearing):
            return "Ongoing hearing"
        elif any(substring.lower() in still_hearing for substring in intermittent_hearing):
            return "Intermittent hearing"
        elif any(substring.lower() in still_hearing for substring in rare_or_never_hearing):
            return "Rarely or no longer hear"
        else:
            return "Unknown"
    except:
        return "Unknown"


df["still_hearing_type"] = df["still_hearing"].apply(categorize_still_hearing)
df.drop(columns=["still_hearing"],
        inplace=True)

# Louder_ear - no changes needed

# ever_Stop
never_stops = ["Never", "Always", "no", "all the time", "permanently", "perpetual",
               "constant", "does not stop", "persistent", "constantly", "nope"]  # If not in this, then it does sometimes stop
can_stop = ["yes", "sometimes", "occasionally", "varies", "every few", "intermittent", "on and off", "stopped", "comes back", "returns",
            "some", "pause", "stopped", "variable", "erratic", "at a time", "does stop"]



def categorize_ever_stops(ever_stops):
    ever_stops = ever_stops.lower()
    try:
        if any(substring.lower() in ever_stops for substring in never_stops):
            return "Never stops"
        elif any(substring.lower() in ever_stops for substring in can_stop):
            return "Can stop"
        else:
            return "Unknown"
    except AttributeError:
        return "Unknown"
df["ever_stops_flag"] = df["ever_stop"].apply(categorize_ever_stops)
df.drop(columns=["ever_stop"],
        inplace=True)


#  noise_freq
def get_hz(noise_freq):
    hz_pattern = r"[0-9]{2,3}"  # 2 or 3 numbers next to each other
    try:
        hz = re.search(hz_pattern, noise_freq).group(0)
    except:
        hz = "Unknown"
    return hz


df["noise_hertz"] = df["noise_freq"].apply(get_hz)
df.drop(columns=["noise_freq"],
        inplace=True)


# pulse_flag
def does_pulse(pulse_flag):
    pulse_lower = pulse_flag.lower()
    yes_flag = ["yes", "sometimes", "similar", "pulse", "oscillat", "pulsing"]
    no_flag = ["no", "constant", "continuous", "does not", "doesn\'t"]
    if any(str(substring).lower() in pulse_lower for substring in yes_flag):
        return "Yes"
    elif any(str(substring).lower() in pulse_lower for substring in no_flag):
        return "No"
    else:
        return "Unknown"



df["does_pulse"] = df["pulse_flag"].apply(does_pulse)
df.drop(columns=["pulse_flag"],
        inplace=True)

# symptom - note this take few minutes to run.
symptoms_list = ["Insomnia", "Anxiety", "Headache", "Nausea", "Sense of vibration", "Ear discomfort", "Annoyance",
                 "Stress"]


def create_symptom_columns(df):
    for symptom in symptoms_list:
        df[symptom] = df["symptoms"].str.contains(symptom.lower(), case=False, na=False).astype(int)



create_symptom_columns(df)  # Creates a few new columns with 0 or 1 o
df = df.rename(columns={"symptoms": "symptoms_full_desc"})
df["symptoms_full_desc"] = df["symptoms_full_desc"].fillna("No description")

# headshake_stop_flag
df.replace({"headshake_stop_flag": {"Yes.1": "yes",
                                    "No": "no",
                                    "nan": "no",
                                    np.nan: "no",
                                    "Not sure": "no",
                                    "Yes": "yes"}},
           inplace=True)

# where_loudest - no change needed

# when_loudest
df.replace({"when_loudest": {"During the day": "Daytime",
                             "During the night": "Nighttime"}},
           inplace=True)

# dominant_hand
df.replace({"dominant_hand": {"I'm right-handed": "Right",
                              "I'm ambidextrous (I can use both hands equally well)": "Ambidextrous",
                              "I'm left-handed": "Left"}},
           inplace=True)

# others_heard_flag - No change required

# can_mask_flag - No change required


# hearing_medical_issue_flag - difficult to process completely accurately
hearing_damage_or_issue = ["tinnitus", "syndrome", "deaf", "worse", "hearing loss", "vertigo", "hearing aid", "ringing",
                           "deafness", "burst eardrum", "chronic"]
definite_no_damage = ["no medical issues", "no health issues", "no tinnitus", "normal", "no damage", "no hearing loss",
                      "none", "good hearing", "NA", "n/a", "nothing diagnosed", "Nil", "no hearing problems"]


def classify_medical_issue(hearing_medical_issue_flag):
    hearing_issue = str(hearing_medical_issue_flag).lower()
    try:
        if any(str(substring).lower() in hearing_issue for substring in hearing_damage_or_issue):
            return "Yes"
        elif any(str(substring).lower() in hearing_issue for substring in definite_no_damage):
            return "No"
        else:
            return "Unknown"
    except [AttributeError, KeyError]:
        return "Unknown"


df["medical_issue"] = df["hearing_medical_issue_flag"].apply(classify_medical_issue)
df["medical_issue"] = df["medical_issue"].fillna("Unknown")
df = df.rename(columns={"hearing_medical_issue_flag": "hearing_issue_desc"})
df["hearing_issue_desc"] = df["hearing_issue_desc"].fillna("No Description")
# medication
not_medicated_list = ["none", "no medication", "no", "no longer", "used to", "nothing", "as a child", "N/A",
                      "not taking", "None", "do not take", "I dont take", "no medications", "Nil"]
currently_medicated_list = ["yes", "I take", "I Currently", "medications for", "pain", "other", "sleeping",
                            "diabetes", "blood pressure", "cholestrol", "azole", "apine", "pills", "medication for",
                            "ine", "naproxen", "aspirin", "nol", "dol", "antidepressant"]


def currently_medicated(medication):
    medication = medication.lower()
    if any(str(substring).lower() in medication for substring in currently_medicated_list):
        return "Currently medicated"
    elif any(str(substring).lower() in medication for substring in not_medicated_list):
        return "No current medication"
    else:
        return "No current medication"


df["currently_medicated"] = df["medication"].apply(currently_medicated)
df["currently_medicated"] = df["currently_medicated"].fillna("No current medication")
df["medication"] = df["medication"].fillna("None")
df = df.rename(columns={"medication": "medication_desc"})
df["medication_desc"] = df["medication_desc"].fillna("No medication description given")

# mental_disorder
df.replace({"mental_disorder": {
    "ADHD (Attention Deficit - Hyperactivity Disorder);Autistic Tendencies, including what was formerly known as \"Asperger\'s\".;Schizophrenia": "ADHD, Autism, Schizophrenia",
    "None of the above;Prefer not to answer": "None",
    "Autistic Tendencies, including what was formerly known as \"Asperger\'s\"., None of the above": "Autism",
    "Prefer not to answer": "None",
    "ADHD (Attention Deficit - Hyperactivity Disorder);Autistic Tendencies, including what was formerly known as \"Asperger\'s\".": "ADHD, Autism",
    "ADHD (Attention Deficit - Hyperactivity Disorder);Schizophrenia": "ADHD, Schizophrenia",
    "None of the above, Prefer not to answer": "None",
    "ADHD (Attention Deficit - Hyperactivity Disorder), Autistic Tendencies, including what was formerly known as \"Asperger\'s\".": "ADHD, Autism",
    "ADHD (Attention Deficit - Hyperactivity Disorder), Autistic Tendencies, including what was formerly known as \"Asperger\'s\"., Schizophrenia": "ADHD, Autism, Schizophrenia",
    "Autistic Tendencies, including what was formerly known as \"Asperger\'s\".;Prefer not to answer": "Autism",
    "None of the above": "None",
    "Schizophrenia": "Schizophrenia",
    "ADHD (Attention Deficit - Hyperactivity Disorder)": "ADHD",
    "Autistic Tendencies, including what was formerly known as \"Asperger\'s\"., Schizophrenia": "ADHD, Schizophrenia",
    "ADHD (Attention Deficit - Hyperactivity Disorder), Schizophrenia": "ADHD, Schizophrenia",
    "ADHD (Attention Deficit - Hyperactivity Disorder), None of the above": "ADHD",
    "nan": "None",
    "ADHD (Attention Deficit - Hyperactivity Disorder), Prefer not to answer": "ADHD",
    "Autistic Tendencies, including what was formerly known as \"Asperger\'s\".;Schizophrenia": "Autism, Schizophrenia",
    "Autistic Tendencies, including what was formerly known as \"Asperger\'s\".": "Autism"}
},
    inplace=True)
df["mental_disorder"] = df["mental_disorder"].fillna("None")

df.replace({"mental_disorder": {np.nan: "None",
                                     "nan": "None"}},
           inplace=True)


# vertigo_balance_flag
df.replace({"vertigo_balance_flag": {"I'm sure I've had it but didn't go to the doctor": "Yes",
                                     np.nan: "No",
                                     "nan": "No",
                                     "No.2": "No"}},
           inplace=True)
df["vertigo_balance_flag"] = df["vertigo_balance_flag"].fillna("No")

# weight_desc
df["weight_desc"] = df["weight_desc"].fillna("Prefer not to answer")

# antibiotic_use - used some from a list online
antibiotics_list = ["Yes", "I take", "currently take", "many", "Gentamicin", "Ertapenem", "Cephalosporin", "Ciprofloxacin",
               "Clindamycin", "Erythromycin", "Metronidazole", "Amoxicillin", "Tetracycline", "imipenem", "meropenem",
               "cefuroxime", "ceftriaxone", "ceftazidime", "levofloxacin", "moxifloxacin", "clarithromycin",
               "azithromycin",
               "benzylpenicillin", "piperacillin", "ticarcillin", "Co-amoxiclav", "flucloxacillin",
               "doxycycline", "minocycline", "fusidic acid", "chloraphenicol", "nitrofuratoin", "trimethoprim"]
never_antibiotics = ["Never", "None at all", "none", "I haven\'t taken", "I dont\'t take any", "nil", "haven\'r taken", "no",]


def use_antibiotics(antibiotic_use):
    antibiotics = antibiotic_use.lower()
    if any(str(substring).lower() in antibiotics for substring in antibiotics_list):
        return "Yes"
    elif any(str(substring).lower() in antibiotics for substring in never_antibiotics):
        return "No"
    else:
        return "No answer"


df["antibiotic_flag"] = df["antibiotic_use"].apply(use_antibiotics)
df["antibiotic_flag"] = df["antibiotic_flag"].fillna("No answer")
df.drop(columns=["antibiotic_use"],
        inplace=True)

# soft_drink_consumption
not_drink_soft_drinks = ["No", "I do not", "I don\'t", "never", "none", "don\t drink it", "do not drink",
                         "do not consume", "Nil", "0"]
does_drink_soft_drinks = ["yes", "regular", "sometimes", "Occasionally", "per month", "/month", "low calorie",
                          "low-calorie", "rarely", "daily", "cans", "at least", "usually", "most days", "weekly", "most"]


def drinks_soft_drinks(soft_drink_consumption):
    softdrinks = soft_drink_consumption.lower()
    if any(str(substring).lower() in softdrinks for substring in not_drink_soft_drinks):
        return "No"
    if any(str(substring).lower() in softdrinks for substring in does_drink_soft_drinks):
        return "Yes"
    else:
        return "Unknown"


df["soft_drinks_flag"] = df["soft_drink_consumption"].apply(drinks_soft_drinks)
df.drop(columns=["soft_drink_consumption"],
        inplace=True)
df["soft_drinks_flag"] = df["soft_drinks_flag"].fillna("Unknown")

# sensitive_sounds_flag
df.replace({"sensitive_sounds_flag": {"Very much so.": "Very sensitive",
                                      "Very much so": "Very sensitive",
                                      "Not especially": "Normal sensitivity",
                                      "Not at all": "Low sensitivity",
                                      "Yes": "Very sensitive",
                                      np.nan: "Normal sensitivity",
                                      None: "Normal sensitivity"}},
           inplace=True)

# location_desc

df.replace({"location_desc": {"Apartment block, condo, or dormitory": "Apartment",
                              "Stand alone house in the city or suburbs": "Urban detached house",
                              "House in an isolated location": "Remote house",
                              np.nan: "Unknown",
                              None: "Unknown"}},
           inplace=True)

# change_with_weather_flag
weather_changes = ["yes", "it does", "does change", "changes", "louder", "quieter", "higher", "lower", ]
weather_not_changes = ["no", "does not change", "doesn\'t change", "not really", "none", "don\'t know", "unsure",
                       "not sure", "never", "not noticed"]


def changes_weather(change_with_weather_flag):
    weather = change_with_weather_flag.lower()
    if any(str(substring).lower() in weather for substring in weather_changes):
        return "Yes"
    if any(str(substring).lower() in weather for substring in weather_not_changes):
        return "No"
    else:
        return "Unknown"


df["changes_weather"] = df["change_with_weather_flag"].apply(changes_weather)
df.drop(columns=["change_with_weather_flag"],
        inplace=True)

# change_with_season_flag
changes_with_season = ["yes", "winter", "summer", "more", "less", "worse"]


def changes_season(change_with_season_flag):
    season = change_with_season_flag.lower()
    if any(str(substring).lower() in season for substring in changes_with_season):
        return "Yes"
    else:
        return "No"


df["changes_season"] = df["change_with_season_flag"].apply(changes_season)
df.drop(columns=["change_with_season_flag"],
        inplace=True)

# high_altitude
no_change_altitude = ["no", "not", "nill", "haven\'t heard", "don\'t know", "can\'t say", "do not know", "n/a", "n.a",
                      "don\'t think", "nothing", "not sure", "unsure", "idk", "not applicable", "nope"]
change_altitude = ["yes", "louder", "queiter", "less", "more", "higher", "disappear"]


def changes_altitude(high_altitude):
    altitude = high_altitude.lower()
    if any(str(substring).lower() in altitude for substring in change_altitude):
        return "Yes"
    else:
        return "No"


df["changes_altitude_flag"] = df["high_altitude"].apply(changes_altitude)
df["changes_altitude_flag"] = df["changes_altitude_flag"].fillna("No")
df = df.rename(columns={"high_altitude": "high_altitude_desc"})
df["high_altitude_desc"] = df["high_altitude_desc"].fillna("No description given")

# heard_elsewhere_flag
also_heard = ["yes", "Y", "never leave", "always hear", "everywhere", "every place", "all over", "anywhere", "also heard", "ja"]
not_heard_elsewhere = ["no", "only here", "nowhere else", "not sure", "unsure", "not really", "not", "only", "haven\'t noticed", "haven\'t", "never", "nope", "non",
                       "don\'t think"]
def heard_elsewhere(heard_elsewhere_flag):
    heard = heard_elsewhere_flag.lower()
    if any(str(substring).lower() in heard for substring in also_heard):
        return "Yes"
    if any(str(substring).lower() in heard for substring in not_heard_elsewhere):
        return "No"
    else:
        return "Unknown"

df["heard_elsewhere"] = df["heard_elsewhere_flag"].apply(heard_elsewhere)
df["heard_elsewhere"] = df["heard_elsewhere"].fillna("Unknown")
df.drop(columns=["heard_elsewhere_flag"],
        inplace=True)

print(set(df["prev_loud_work_flag"]))
loud_work_list = ["yes", "music", "construction", "factory", "manufactur", "plant", "factories", "woodwork", "metal", "drum", "heavy", "mill", "army", "military",
                  "nightclub", "woodwork", "workshop", "in the past", "acoustic", "lawn", "train", "airport", "teacher", "farm", "kitchen", "engineer", "band","somewhat",
                  "used to", "machine", "artillery", "nuclear", "power station", "navy", "aircraft", "hearing protection", "DJ", "club", "ear protection", "earplug", "builder",
                  "building", "oil", "mine", "truck", "protection", "protect"]
def in_loud_work(prev_loud_work_flag):
    loud_work = prev_loud_work_flag.lower()
    if any(str(substring).lower() in loud_work for substring in loud_work_list):
        return "Yes"
    else:
        return "No"
df["loud_work"] = df["prev_loud_work_flag"].apply(in_loud_work)
df["loud_work"] = df["loud_work"].fillna("No")
df.drop(columns=["prev_loud_work_flag"],
        inplace=True)

# Geographic band
def get_latitude_band(latitude):
    if latitude == 0 or latitude == 0.0:
        return "Unknown"
    if -90 < latitude <= -45:
        return "-90 to -45"
    elif -45 < latitude <= -20:
        return "-44 to -20"
    elif -20 < latitude <= 0:
        return "-20 to 0"
    elif -0 < latitude <= 20:
        return "0 to 20"
    elif 20 < latitude <= 45:
        return "20 to 45"
    elif 45 < latitude <=90:
        return "45 to 90"
    else:
        return "Unknown"
df["latitude_band"] = df["latitude"].apply(get_latitude_band)




# Write out the table as a CSV
df.to_csv("C:/Users/tomwr/Datascience/Datasets/Tabular/World Hum Map/hummap_processed.csv")
