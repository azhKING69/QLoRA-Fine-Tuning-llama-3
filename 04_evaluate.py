import pandas as pd
import json
import subprocess

# --- CONFIGURATION ---
EVAL_DATA_FILE = "gold_eval_set.csv"
MODEL_NAME = "osha-1k-model" # The name registered in Ollama
OUTPUT_FILE = "predictions.csv"
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

def query_ollama(narrative):
    prompt = f"Narrative: {narrative}"
    process = subprocess.Popen(
        ['ollama', 'run', MODEL_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    stdout, stderr = process.communicate(input=full_prompt)
    return stdout.strip()

if __name__ == "__main__":
    print(f"Loading Evaluation Set: {EVAL_DATA_FILE}")
    try:
        df = pd.read_csv(EVAL_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {EVAL_DATA_FILE} not found. Run 01_data_prep.py first.")
        exit(1)
        
    results = []
    correct = 0
    total = 0
    
    print(f"Starting grading using local Ollama model: {MODEL_NAME}")
    
    for i, row in df.iterrows():
        raw_output = query_ollama(row['Final Narrative'])
        result_row = {"ID": row.get("ID", i), "Narrative": row["Final Narrative"], "Raw_LLM_Output": raw_output}
        
        try:
            prediction = json.loads(raw_output)
            fields = [("event_type", "TRUE_event"), ("injury_nature", "TRUE_nature"), 
                      ("body_part", "TRUE_body"), ("source_equipment", "TRUE_source")]
            
            for pred_key, true_col in fields:
                if true_col in row:
                    total += 1
                    
                    # Get the ground truth answer (fallback to mapped_ if human hasn't filled TRUE_ yet)
                    true_val = str(row[true_col]).strip().lower()
                    if true_val == "" or true_val == "nan":
                        fallback_col = true_col.replace("TRUE_", "mapped_")
                        if fallback_col in row:
                            true_val = str(row[fallback_col]).strip().lower()
                            
                    pred_val = str(prediction.get(pred_key, "")).strip().lower()
                    result_row[f"pred_{pred_key}"] = prediction.get(pred_key, "")
                    
                    if pred_val == true_val:
                        correct += 1
        except Exception:
            if "TRUE_event" in row:
                total += 4
            result_row["pred_event"] = "JSON_ERROR"
            
        results.append(result_row)
        print(f"Graded {i+1}/{len(df)}...")
        
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n========================================")
        print(f"Final Local Model Accuracy: {accuracy:.1f}%")
        print(f"========================================")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")
