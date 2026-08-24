import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAW_DATA_FILE = "dataset.csv"
CLEAN_TRAIN_POOL = "train_pool.csv"
GOLD_EVAL_FILE = "gold_eval_set.csv"
# ---------------------

def map_event_type(raw):
    raw = str(raw).lower()
    if pd.isna(raw): return "other"
    if "fall" in raw and "lower level" in raw: return "fall_to_lower_level"
    if "fall" in raw and "same level" in raw: return "fall_same_level"
    if "fall" in raw and ("surface" in raw or "through" in raw): return "fall_through_surface"
    if "struck by" in raw and ("vehicle" in raw or "truck" in raw): return "struck_by_vehicle"
    if "struck by" in raw: return "struck_by_object"
    if "struck against" in raw: return "struck_against"
    if "caught in" in raw or "compressed" in raw: return "caught_in_between"
    if "electrocution" in raw or "electricity" in raw: return "electrocution"
    if "heat" in raw or "environmental" in raw: return "heat_exposure"
    if "trench" in raw or "cave-in" in raw: return "trench_cave_in"
    if "explosion" in raw or "fire" in raw: return "explosion_ignition"
    if "chemical" in raw or "exposure" in raw: return "chemical_exposure"
    if "overturned" in raw: return "equipment_overturned"
    if "contact" in raw and "hot" in raw: return "contact_hot_substance"
    if "collapse" in raw or "engulfment" in raw: return "collapse_engulfment"
    return "other"

def map_injury_nature(raw):
    raw = str(raw).lower()
    if pd.isna(raw): return "other"
    if "fracture" in raw: return "fracture"
    if "amputation" in raw: return "amputation"
    if "laceration" in raw or "cut" in raw: return "laceration"
    if "contusion" in raw or "bruise" in raw: return "contusion_pain"
    if "burn" in raw: return "burn"
    if "sprain" in raw or "strain" in raw or "tear" in raw: return "sprain_strain_tear"
    if "concussion" in raw or "intracranial" in raw: return "intracranial"
    if "internal" in raw: return "internal_injury"
    if "crushing" in raw: return "crushing"
    if "heat" in raw: return "heat_illness"
    if "dislocation" in raw: return "dislocation"
    if "poisoning" in raw or "toxic" in raw: return "poisoning_toxic"
    if "heart attack" in raw: return "heart_attack"
    return "other"

def map_body_part(raw):
    raw = str(raw).lower()
    if pd.isna(raw): return "other"
    if "finger" in raw or "hand" in raw: return "finger_hand"
    if "leg" in raw or "knee" in raw: return "leg"
    if "foot" in raw or "ankle" in raw or "toe" in raw: return "foot_ankle"
    if "arm" in raw or "elbow" in raw: return "arm"
    if "wrist" in raw: return "wrist"
    if "shoulder" in raw: return "shoulder"
    if "neck" in raw: return "neck"
    if "head" in raw or "brain" in raw or "face" in raw: return "head_brain"
    if "eye" in raw: return "eye"
    if "back" in raw or "spine" in raw: return "back_spine"
    if "chest" in raw or "trunk" in raw: return "chest_trunk"
    if "hip" in raw or "pelvis" in raw: return "hip_pelvis"
    if "system" in raw: return "body_systems"
    if "multiple" in raw: return "multiple"
    return "other"

def map_source(raw):
    raw = str(raw).lower()
    if pd.isna(raw): return "other"
    if "ladder" in raw: return "ladder"
    if "scaffold" in raw: return "scaffold"
    if "roof" in raw: return "roof"
    if "aerial" in raw or "lift" in raw: return "aerial_lift"
    if "floor" in raw or "ground" in raw or "stair" in raw: return "floor_ground_stair"
    if "window" in raw or "opening" in raw: return "window_opening"
    if "vehicle" in raw or "truck" in raw or "forklift" in raw: return "vehicle_heavy_equip"
    if "tool" in raw or "saw" in raw or "grinder" in raw: return "power_tool"
    if "machinery" in raw or "equipment" in raw: return "machinery_equipment"
    if "electrical" in raw or "wire" in raw: return "electrical_source"
    if "pipe" in raw or "duct" in raw: return "pipe_duct"
    if "rebar" in raw: return "rebar"
    if "metal" in raw or "steel" in raw: return "metal_material"
    if "wood" in raw or "building material" in raw: return "building_materials_parts"
    if "nail" in raw or "fastener" in raw: return "nail_fastener"
    if "trench" in raw or "excavation" in raw: return "trench_excavation"
    if "debris" in raw: return "debris"
    if "pole" in raw: return "pole"
    if "structural" in raw: return "structural"
    if "environmental" in raw or "weather" in raw: return "environmental"
    if "bodily motion" in raw: return "bodily_motion"
    if "box" in raw or "container" in raw: return "boxes_containers"
    if "tank" in raw or "vat" in raw: return "tanks_vats"
    if "cover" in raw or "lid" in raw: return "covers_lids"
    if "chemical" in raw or "fume" in raw: return "chemicals_fumes"
    if "hose" in raw or "cable" in raw: return "hose_cable"
    return "other"

if __name__ == "__main__":
    print(f"Loading raw data from {RAW_DATA_FILE}...")
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Could not find {RAW_DATA_FILE}. Please ensure it is in the root directory.")
        exit(1)
        
    print(f"Total raw records: {len(df_raw)}")
    
    # Filter for construction (NAICS 23)
    df_const = df_raw[df_raw['Primary NAICS'].astype(str).str.startswith('23')].dropna(subset=['Final Narrative']).copy()
    print(f"Construction records after filtering: {len(df_const)}")
    
    # Apply Mapping
    print("Applying V3 weak supervision mappings...")
    df_const['mapped_event'] = df_const['EventTitle'].apply(map_event_type)
    df_const['mapped_nature'] = df_const['NatureTitle'].apply(map_injury_nature)
    df_const['mapped_body'] = df_const['Part of Body Title'].apply(map_body_part)
    df_const['mapped_source'] = df_const['SourceTitle'].apply(map_source)
    df_const['mapped_hospitalized'] = df_const['Hospitalized'].astype(str).str.lower().apply(lambda x: True if x == 'yes' else False)
    df_const['mapped_amputation'] = df_const['Amputation'].astype(str).str.lower().apply(lambda x: True if x == 'yes' else False)
    
    # Extract Gold Set (Holdout 200)
    print(f"Generating {GOLD_EVAL_FILE}...")
    df_gold = df_const.sample(n=200, random_state=42).copy()
    
    # Keep ID, Narrative, and the automated mapped columns
    df_gold = df_gold[['ID', 'Final Narrative', 'mapped_event', 'mapped_nature', 'mapped_body', 'mapped_source', 'mapped_hospitalized', 'mapped_amputation']]
    
    # Rename boolean columns to match the screenshot
    df_gold.rename(columns={'mapped_hospitalized': 'mapped_hosp', 'mapped_amputation': 'mapped_amp'}, inplace=True)
    
    # Insert empty TRUE_ columns right next to their corresponding mapped columns for easy side-by-side human grading
    df_gold.insert(df_gold.columns.get_loc('mapped_event') + 1, 'TRUE_event', '')
    df_gold.insert(df_gold.columns.get_loc('mapped_nature') + 1, 'TRUE_nature', '')
    df_gold.insert(df_gold.columns.get_loc('mapped_body') + 1, 'TRUE_body', '')
    df_gold.insert(df_gold.columns.get_loc('mapped_source') + 1, 'TRUE_source', '')
    df_gold.insert(df_gold.columns.get_loc('mapped_hosp') + 1, 'TRUE_hosp', '')
    df_gold.insert(df_gold.columns.get_loc('mapped_amp') + 1, 'TRUE_amp', '')
        
    df_gold.to_csv(GOLD_EVAL_FILE, index=False)
    
    # Export Train Pool (Excluding Gold Set)
    print(f"Generating {CLEAN_TRAIN_POOL}...")
    df_train = df_const.drop(df_gold.index)
    
    final_cols = ['Final Narrative', 'mapped_event', 'mapped_nature', 'mapped_body', 'mapped_source', 'mapped_hospitalized', 'mapped_amputation']
    df_train = df_train[[c for c in final_cols if c in df_train.columns]]
    df_train.columns = ['Narrative', 'event_type', 'injury_nature', 'body_part', 'source_equipment', 'hospitalized', 'amputation']
    df_train.to_csv(CLEAN_TRAIN_POOL, index=False)
    
    print("Data Prep Complete! Files saved to root directory.")
