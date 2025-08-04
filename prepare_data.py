import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

# Load the datasets
workorder_df = pd.read_csv('maintenance dataset/preprocessed_workorder.csv')
partorder_df = pd.read_csv('maintenance dataset/preprocessed_partorder.csv')
co_replaced_df = pd.read_csv('maintenance dataset/filtered_co_replaced_parts.csv')
sparepart_df = pd.read_csv('maintenance dataset/sparepart_list.csv')

# --- Data Cleaning: Standardize Column Names ---
# Replace spaces with underscores and convert to lowercase for easier access
workorder_df.columns = workorder_df.columns.str.replace(' ', '_').str.lower()
partorder_df.columns = partorder_df.columns.str.replace(' ', '_').str.lower()
co_replaced_df.columns = co_replaced_df.columns.str.replace(' ', '_').str.lower()
sparepart_df.columns = sparepart_df.columns.str.replace(' ', '_').str.lower()

# Rename columns for consistency
partorder_df.rename(columns={'work_order': 'workorder_id', 'part': 'part_number'}, inplace=True)
co_replaced_df.rename(columns={'part_1': 'part_1', 'part_2': 'part_2'}, inplace=True)
sparepart_df.rename(columns={'standarized_part_name': 'part_number'}, inplace=True)


print("Workorder data shape:", workorder_df.shape)
print("Partorder data shape:", partorder_df.shape)
print("Co-replaced parts data shape:", co_replaced_df.shape)
print("Sparepart list shape:", sparepart_df.shape)

# --- Entity Mapping ---
# Get unique entities from each dataframe
vehicles = workorder_df['frame_serial_no'].astype(str).unique()
spare_parts = sparepart_df['part_number'].astype(str).unique()
ordered_parts = partorder_df['part_number'].astype(str).unique()
parts = np.unique(np.concatenate([spare_parts, ordered_parts]))
workorders = workorder_df['work_order_no'].astype(str).unique()
subsystems = sparepart_df['subsystem'].astype(str).unique()
base_models = workorder_df['base_model'].astype(str).unique()
vehicle_ages = workorder_df['vehicle_age'].astype(str).unique()

# Create a single list of all entities
all_entities = np.concatenate([vehicles, parts, workorders, subsystems, base_models, vehicle_ages])
all_entities = np.unique(all_entities)

# Create entity to ID mapping
entity2id = {entity: i for i, entity in enumerate(all_entities)}
n_entities = len(entity2id)

# --- Relation Mapping ---
relations = [
    'underwent_service',
    'replaced_part',
    'is_part_of',
    'co_replaced_with',
    'is_model',
    'at_age'
]
relation2id = {relation: i for i, relation in enumerate(relations)}
n_relations = len(relation2id)

print(f"Number of unique entities: {n_entities}")
print(f"Number of unique relations: {n_relations}")

# Create the output directory if it doesn't exist
output_dir = 'dataset/toyota_maintenance'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save entity and relation lists
entity_list_df = pd.DataFrame(entity2id.items(), columns=['name', 'id'])
entity_list_df.to_csv(os.path.join(output_dir, 'entity_list.txt'), sep='\t', header=False, index=False)

relation_list_df = pd.DataFrame(relation2id.items(), columns=['name', 'id'])
relation_list_df.to_csv(os.path.join(output_dir, 'relation_list.txt'), sep='\t', header=False, index=False)

print("Saved entity_list.txt and relation_list.txt")

kg_triples = []

# 1. vehicle --[underwent_service]--> workorder
for _, row in workorder_df.iterrows():
    vehicle_id = entity2id[row['frame_serial_no']]
    workorder_id = entity2id[row['work_order_no']]
    relation_name = 'underwent_service'
    kg_triples.append((vehicle_id, relation_name, workorder_id))

# 2. vehicle --[replaced_part]--> part
vehicle_part_df = pd.merge(workorder_df[['work_order_no', 'frame_serial_no']], partorder_df[['workorder_id', 'part_number']], left_on='work_order_no', right_on='workorder_id')
for _, row in vehicle_part_df.iterrows():
    vehicle_id = entity2id[row['frame_serial_no']]
    part_id = entity2id[row['part_number']]
    relation_name = 'replaced_part'
    kg_triples.append((vehicle_id, relation_name, part_id))

# 3. part --[is_part_of]--> subsystem
for _, row in sparepart_df.iterrows():
    part_id = entity2id[row['part_number']]
    subsystem_id = entity2id[row['subsystem']]
    relation_name = 'is_part_of'
    kg_triples.append((part_id, relation_name, subsystem_id))

# 4. part --[co_replaced_with]--> part
for _, row in co_replaced_df.iterrows():
    part1_id = entity2id[row['part_1']]
    part2_id = entity2id[row['part_2']]
    relation_name = 'co_replaced_with'
    kg_triples.append((part1_id, relation_name, part2_id))
    kg_triples.append((part2_id, relation_name, part1_id)) # Symmetric relation

# 5. vehicle --[is_model]--> base_model
for _, row in workorder_df.iterrows():
    vehicle_id = entity2id[row['frame_serial_no']]
    model_id = entity2id[row['base_model']]
    relation_name = 'is_model'
    kg_triples.append((vehicle_id, relation_name, model_id))

# 6. vehicle --[at_age]--> vehicle_age
for _, row in workorder_df.iterrows():
    vehicle_id = entity2id[row['frame_serial_no']]
    age_id = entity2id[row['vehicle_age']]
    relation_name = 'at_age'
    kg_triples.append((vehicle_id, relation_name, age_id))

# Save the KG
kg_df = pd.DataFrame(kg_triples, columns=['head', 'relation', 'tail'])
# Replace relation names with IDs for the final kg.txt
kg_df['relation'] = kg_df['relation'].map(relation2id)
kg_df.to_csv(os.path.join(output_dir, 'kg.txt'), sep='\t', header=False, index=False)

print(f"Generated and saved kg.txt with {len(kg_df)} triples.")

# Group parts by workorder to create sessions
partorder_df['part_id'] = partorder_df['part_number'].map(entity2id)
sessions = partorder_df.groupby('workorder_id')['part_id'].apply(list).values.tolist()

# Split data into training and testing sets (80/20)
train_sessions, test_sessions = train_test_split(sessions, test_size=0.2, random_state=42)

# Save the datasets using pickle
with open(os.path.join(output_dir, 'train.txt'), 'wb') as f:
    pickle.dump(train_sessions, f)

with open(os.path.join(output_dir, 'test.txt'), 'wb') as f:
    pickle.dump(test_sessions, f)

print(f"Generated and saved train.txt ({len(train_sessions)} sessions) and test.txt ({len(test_sessions)} sessions).")