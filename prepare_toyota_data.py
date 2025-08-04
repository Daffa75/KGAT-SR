import os
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = 'maintenance dataset/'
OUTPUT_DIR = 'datasets/toyota_maintenance/'

# Input files
WORKORDER_FILE = os.path.join(DATA_DIR, 'preprocessed_workorder.csv')
PARTORDER_FILE = os.path.join(DATA_DIR, 'preprocessed_partorder.csv')
CO_REPLACED_FILE = os.path.join(DATA_DIR, 'filtered_co_replaced_parts.csv')
SPAREPART_FILE = os.path.join(DATA_DIR, 'sparepart_list.csv')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
workorder_df = pd.read_csv(WORKORDER_FILE)
partorder_df = pd.read_csv(PARTORDER_FILE)
co_replaced_df = pd.read_csv(CO_REPLACED_FILE)
sparepart_df = pd.read_csv(SPAREPART_FILE)
print("Data loaded.")

# --- Initialize KG components ---
entity_set = set()
relation_set = set()
kg_triples = []

# --- 1. vehicle `underwent_service` workorder ---
print("Processing 'underwent_service' relations...")
relation = 'underwent_service'
relation_set.add(relation)
for _, row in workorder_df.iterrows():
    vehicle = str(row['Frame Serial No'])
    work_order = str(row['Work Order No'])
    entity_set.add(vehicle)
    entity_set.add(work_order)
    kg_triples.append((vehicle, relation, work_order))

# --- 2. workorder `replaced_part` part ---
print("Processing 'replaced_part' relations...")
relation = 'replaced_part'
relation_set.add(relation)
for _, row in partorder_df.iterrows():
    work_order = str(row['Work Order'])
    part = str(row['standarized_part_name'])
    entity_set.add(work_order)
    entity_set.add(part)
    kg_triples.append((work_order, relation, part))

# --- 3. part `is_part_of` subsystem ---
print("Processing 'is_part_of' relations...")
relation = 'is_part_of'
relation_set.add(relation)
# Create a mapping from part name to subsystem from the sparepart list
part_to_subsystem = sparepart_df.set_index('standarized_part_name')['subsystem'].to_dict()
for part, subsystem in part_to_subsystem.items():
    if pd.notna(part) and pd.notna(subsystem):
        part_str = str(part)
        subsystem_str = str(subsystem)
        entity_set.add(part_str)
        entity_set.add(subsystem_str)
        kg_triples.append((part_str, relation, subsystem_str))

# --- 4. part `co_replaced_with` part ---
print("Processing 'co_replaced_with' relations...")
relation = 'co_replaced_with'
relation_set.add(relation)
for _, row in co_replaced_df.iterrows():
    part1 = str(row['Part 1'])
    part2 = str(row['Part 2'])
    entity_set.add(part1)
    entity_set.add(part2)
    kg_triples.append((part1, relation, part2))
    # The relation is symmetric
    kg_triples.append((part2, relation, part1))

# --- 5. vehicle `is_model` basemodel ---
print("Processing 'is_model' relations...")
relation = 'is_model'
relation_set.add(relation)
for _, row in workorder_df.iterrows():
    vehicle = str(row['Frame Serial No'])
    base_model = str(row['Base Model'])
    entity_set.add(vehicle)
    entity_set.add(base_model)
    kg_triples.append((vehicle, relation, base_model))

# --- 6. vehicle `at_age` vehicle_age ---
print("Processing 'at_age' relations...")
relation = 'at_age'
relation_set.add(relation)
for _, row in workorder_df.iterrows():
    vehicle = str(row['Frame Serial No'])
    age = str(row['Vehicle Age'])
    entity_set.add(vehicle)
    entity_set.add(age)
    kg_triples.append((vehicle, relation, age))

print(f"Total triples in KG: {len(kg_triples)}")
print(f"Total unique entities: {len(entity_set)}")
print(f"Total unique relations: {len(relation_set)}")

# --- Create Mappings ---
entity_list = sorted(list(entity_set))
relation_list = sorted(list(relation_set))

entity2id = {e: i for i, e in enumerate(entity_list)}
relation2id = {r: i for i, r in enumerate(relation_list)}

# --- Save Mappings ---
print("Saving entity and relation lists...")
with open(os.path.join(OUTPUT_DIR, 'entity_list.txt'), 'w') as f:
    for entity in entity_list:
        f.write(f"{entity}\n")

with open(os.path.join(OUTPUT_DIR, 'relation_list.txt'), 'w') as f:
    for i, relation in enumerate(relation_list):
        f.write(f"{relation}\t{i}\n")

# --- Save KG ---
print("Saving knowledge graph...")
with open(os.path.join(OUTPUT_DIR, 'kg.txt'), 'w') as f:
    for h, r, t in kg_triples:
        head_id = entity2id[h]
        relation_id = relation2id[r]
        tail_id = entity2id[t]
        f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")

# --- Process Session Data ---
print("Processing session data...")
# A session is a list of parts replaced in a single work order
sessions = partorder_df.groupby('Work Order')['standarized_part_name'].apply(list)

# Convert part names to their corresponding entity IDs
part_entity_ids = []
for session in sessions:
    # Filter out parts that might not be in the final entity list (if any)
    # and convert to string to match entity set keys
    part_ids = [entity2id[str(part)] for part in session if str(part) in entity2id]
    if len(part_ids) > 1: # Only keep sessions with more than one item
        part_entity_ids.append(part_ids)

print(f"Total sessions created: {len(part_entity_ids)}")

# --- Split and Save Session Data ---
print("Splitting and saving train/test sets...")

# Create sequences and targets
sequences = []
targets = []
for session in part_entity_ids:
    # The model expects a sequence to predict the next item.
    # We'll use all but the last item as input, and the last item as the target.
    if len(session) > 1:
        sequences.append(session[:-1])
        targets.append(session[-1])

print(f"Total sequences for training/testing: {len(sequences)}")

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# The model expects the data in a tuple format: (sequences, targets)
train_data = (train_x, train_y)
test_data = (test_x, test_y)

with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'wb') as f:
    pickle.dump(train_data, f)

with open(os.path.join(OUTPUT_DIR, 'test.txt'), 'wb') as f:
    pickle.dump(test_data, f)

print("Script finished successfully.")
print(f"Output files are saved in '{OUTPUT_DIR}'")