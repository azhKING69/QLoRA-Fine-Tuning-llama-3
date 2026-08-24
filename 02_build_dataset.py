import pandas as pd
import json

# --- CONFIGURATION ---
INPUT_CSV = "train_pool.csv"
SYSTEM_PROMPT = """You are a construction safety analyst. Given an OSHA incident narrative, extract the following structured fields.
Respond with ONLY a valid JSON object. Do not include markdown formatting or explanation.

Fields:
- event_type: one of [fall_to_lower_level, struck_by_object, caught_in_between, struck_by_vehicle, electrocution, heat_exposure, fall_same_level, struck_against, fall_through_surface, contact_hot_substance, collapse_engulfment, trench_cave_in, explosion_ignition, chemical_exposure, equipment_overturned, other]
- injury_nature: one of [fracture, amputation, laceration, contusion_pain, burn, internal_injury, intracranial, crushing, heat_illness, dislocation, poisoning_toxic, sprain_strain_tear, heart_attack, other]
- body_part: one of [finger_hand, leg, head_brain, chest_trunk, multiple, foot_ankle, arm, back_spine, wrist, hip_pelvis, shoulder, body_systems, eye, neck, other]
- source_equipment: one of [ladder, scaffold, roof, power_tool, vehicle_heavy_equip, aerial_lift, electrical_source, structural, floor_ground_stair, pipe_duct, environmental, trench_excavation, nail_fastener, metal_material, rebar, debris, pole, machinery_equipment, building_materials_parts, bodily_motion, boxes_containers, tanks_vats, covers_lids, chemicals_fumes, window_opening, hose_cable, other]
- hospitalized: true or false
- amputation: true or false"""
# ---------------------

def create_chat_format(row):
    target_json = {
        "event_type": row["event_type"],
        "injury_nature": row["injury_nature"],
        "body_part": row["body_part"],
        "source_equipment": row["source_equipment"],
        "hospitalized": bool(row["hospitalized"]),
        "amputation": bool(row["amputation"])
    }
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Narrative: {row['Narrative']}"},
            {"role": "assistant", "content": json.dumps(target_json)}
        ]
    }

if __name__ == "__main__":
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Run 01_data_prep.py first.")
        exit(1)
        
    print("Shuffling dataset to ensure random distribution...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Formatting to ChatML JSON...")
    all_formatted_data = df.apply(create_chat_format, axis=1).tolist()
    
    sizes = {"1k": 1000, "3k": 3000, "5k": 5000}
    
    for name, size in sizes.items():
        subset = all_formatted_data[:size] 
        filename = f"train_{name}.jsonl"
        with open(filename, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item) + "\n")
        print(f"Successfully generated strictly nested dataset: {filename} ({len(subset)} records).")
        
    print("Dataset generation complete!")
