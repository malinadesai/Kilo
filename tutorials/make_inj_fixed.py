import json
import numpy as np

dir_name = 'test_lc_sup/'
file_name = dir_name + 'injection_test_var.json'
new_file_name = dir_name + 'injection_test_fix.json'

with open(file_name, "r") as f:
    data = json.load(f)

new_distances = np.full(
        shape=len(data['injections']['content']['luminosity_distance']),
        fill_value=50,
        )

new_times = np.full(
        shape=len(data['injections']['content']['timeshift']),
        fill_value=0,
        )

# Replace the distance data
data['injections']['content']['luminosity_distance'] = new_distances.tolist()
data['injections']['content']['timeshift'] = new_times.tolist()

# Save the modified data
with open(new_file_name, "w") as f:
    json.dump(data, f, indent=4)
~                                          
