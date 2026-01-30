import os
import yaml
import json
import pandas as pd

# Model to process
model = 'Qwen/Qwen2.5-7B-Instruct'

# Directory containing results
results_dir = 'results/'

# Initialize a list to hold the data for the table
data = []

# Iterate through each subfolder in the results directory
for subfolder in os.listdir(results_dir):
    subfolder_path = os.path.join(results_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Load config.yaml
        config_path = os.path.join(subfolder_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            # Skip if model field doesn't match
            if config.get('model') != model:
                continue
            press_name = config.get('press_name')
            data_dir = config.get('data_dir')

        # Load metrics.json
        metrics_path = os.path.join(subfolder_path, 'metrics.json')
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)
            accuracy = metrics.get('all', 0)

        # Append the data to the list
        data.append({'press_name': press_name, 'data_dir': data_dir, 'accuracy': accuracy})

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Pivot the DataFrame to have press_name as rows and data_dir as columns
pivot_df = df.pivot(index='press_name', columns='data_dir', values='accuracy')

# Calculate the average across all data_dir columns for each press_name
pivot_df['average'] = pivot_df.mean(axis=1)

# Specify the desired column order
desired_order = ['qasper_e', 'multifieldqa_en_e', 'hotpotqa_e', '2wikimqa_e', 'gov_report_e', 'multi_news_e', 
                 'trec_e', 'triviaqa_e', 'samsum_e', 'passage_count_e', 'passage_retrieval_en_e', 'lcc_e', 'repobench-p_e', 'average']

# Reorder columns (only include columns that exist in the dataframe)
available_cols = [col for col in desired_order if col in pivot_df.columns]
pivot_df = pivot_df[available_cols]

# Sort rows by average score in decreasing order
pivot_df = pivot_df.sort_values('average', ascending=True)

# Reset index 
pivot_df = pivot_df.reset_index()

# Make no_press the first row if it exists
if 'no_press' in pivot_df['press_name'].values:
    no_press_row = pivot_df[pivot_df['press_name'] == 'no_press']
    pivot_df = pivot_df[pivot_df['press_name'] != 'no_press']
    pivot_df = pd.concat([no_press_row, pivot_df], ignore_index=True)

# Rename columns to shorter display names
column_mapping = {
    'press_name': 'Method',
    'qasper_e': 'qasper',
    'multifieldqa_en_e': 'multifield',
    'hotpotqa_e': 'hotpot',
    '2wikimqa_e': '2wiki',
    'gov_report_e': 'gov',
    'multi_news_e': 'multinews',
    'trec_e': 'trec',
    'triviaqa_e': 'trivia',
    'samsum_e': 'samsum',
    'passage_count_e': 'p.count',
    'passage_retrieval_en_e': 'p.ret',
    'lcc_e': 'lcc',
    'repobench-p_e': 'repo-p',
    'average': 'average'
}
pivot_df = pivot_df.rename(columns=column_mapping)

# Display results
print(pivot_df.to_string())

# Map press_name values to display names
def map_press_name(name):
    press_name_mapping = {
        'no_press': 'Exact',
        'balance_kv': 'BalanceKV',
        'snapkv': 'SnapKV',
        'streaming_llm': 'StreamingLLM',
        'pyramidkv': 'PyramidKV',
        'uniform': 'Uniform'
    }
    
    # Check if it's in the mapping
    if name in press_name_mapping:
        return press_name_mapping[name]
    
    # Check if it starts with compress_kv_
    if name.startswith('compress_kv_'):
        number = name.replace('compress_kv_', '')
        return f'CompressKV'
    
    return name
# Compute mapped names
mapped_methods = pivot_df['Method'].map(map_press_name)

# Keep only rows where a rename occurred (mapped differs from original)
pivot_df = pivot_df[mapped_methods != pivot_df['Method']].copy()

# Assign mapped Method
pivot_df['Method'] = mapped_methods[mapped_methods.index]

# Filter to keep only the last CompressKV row
compresskv_indices = pivot_df[pivot_df['Method'].str.startswith('CompressKV')].index.tolist()
rows_to_drop = compresskv_indices[:-1] if len(compresskv_indices) > 1 else []
pivot_df = pivot_df.drop(rows_to_drop)

# Find the max value in the average column (excluding Exact row)
pivot_df_for_max = pivot_df[pivot_df['Method'] != 'Exact']
max_avg = pivot_df_for_max['average'].max()

# Apply bold formatting to max average value
def format_value(val, is_max):
    if is_max:
        return f'\\textbf{{{val:.2f}}}'
    return f'{val:.2f}'

pivot_df['average'] = pivot_df.apply(
    lambda row: format_value(row['average'], row['average'] == max_avg and row['Method'] != 'Exact'),
    axis=1
)

# Bold Method entries and headers
pivot_df['Method'] = pivot_df['Method'].apply(lambda x: f'\\textbf{{{x}}}')
bold_headers = {col: f'\\textbf{{{col}}}' for col in pivot_df.columns}
pivot_df_bold = pivot_df.rename(columns=bold_headers)
pivot_df_bold.columns.name = None

# Output the DataFrame in LaTeX format with 2 decimal places
# First column (Method) is center-aligned, all others are center-aligned
# Specify space between columns and on edges
column_format = '@{\\hspace{2.5pt}}c' + '@{\\hspace{4pt}}c' * (len(pivot_df_bold.columns) - 1) + '@{\\hspace{2.5pt}}'

latex_output = pivot_df_bold.to_latex(
    index=False,
    float_format=lambda x: f'{x:.2f}',
    column_format=column_format,
    escape=False,
)

# Add space between rows, but not after header or final row
lines = latex_output.split('\n')
for i, line in enumerate(lines):
    # Add space to rows that end with \\ but exclude header and final row
    if line.rstrip().endswith('\\\\'):
        # Skip the header row (first \\ after toprule) and rows before bottomrule
        if i > 0 and 'toprule' not in lines[i-1] and (i+1 < len(lines) and 'bottomrule' not in lines[i+1]):
            lines[i] = line.rstrip()[:-2] + ' \\\\[.5mm]'

latex_output = '\n'.join(lines)

print(latex_output)