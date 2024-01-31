import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
import numpy as np
from numpy.polynomial.polynomial import Polynomial


# Load data from JSON files
file_path1 = 'h-s_result.json'
file_path2 = 'h-non-s_result.json'
file_path4 = 'non-h-s_result.json'
file_path3 = 'non-h-non-s_result.json'

# Paths to the JSON files
file_paths = [file_path1, file_path2, file_path3, file_path4]
experiment_name = {
    file_path1: "HMR-full",
    file_path2: "HMR-limited",
    file_path3: "NonHMR-full",
    file_path4: "NonHMR-limited"
}

# Function to smooth the data using spline interpolation
def smooth_data(x, y):
    spline = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
    x_smooth = np.linspace(x.min(), x.max(), 100)  # Adjust the number of points for smoothing
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Create a plot
plt.figure(figsize=(12, 8))

# Process and plot each file's data
for file_path in file_paths:
    data = pd.read_json(file_path, lines=True)
    #x_smooth, y_smooth = smooth_data(data['_step'], data['success_rate'])
    x_smooth, y_smooth = data['_step'], data['success_rate']
    plt.plot(x_smooth, y_smooth, label=experiment_name[file_path])  # Use file name as label or customize it

# Format x-axis to display large numbers in a more readable format
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylim(bottom=0)
plt.grid(True)
# Adding labels and title
plt.xlabel('Step')
plt.ylabel('Success Rate')
plt.legend()

# show the plot
plt.savefig('success_rate.png', dpi=1200)
plt.show()
'''
data1 = pd.read_json(file_path1, lines=True)
data2 = pd.read_json(file_path2, lines=True)


# Function to smooth the data using spline interpolation
def smooth_data(x, y):
    spline = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
    x_smooth = np.linspace(x.min(), x.max(), 100)  # Reduce the number of points for less smoothing
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Smooth the data
x1_smooth, y1_smooth = smooth_data(data1['_step'], data1['success_rate'])
x2_smooth, y2_smooth = smooth_data(data2['_step'], data2['success_rate'])

# Create a plot
plt.figure(figsize=(10, 6))

# Plot smoothed data
plt.plot(x1_smooth, y1_smooth, label='File 1')
plt.plot(x2_smooth, y2_smooth, label='File 2')

# Format x-axis to display large numbers in a more readable format
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adding labels and title
plt.xlabel('Step')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Steps (Spline Interpolation)')
plt.legend()

# Show the plot
plt.show()
'''

'''
# Create a plot
plt.figure(figsize=(10, 10))

# Plot data from both files
plt.plot(data1['_step'], data1['success_rate'], label='File 1')
plt.plot(data2['_step'], data2['success_rate'], label='File 2')

# Format x-axis to display large numbers in a more readable format
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adding labels and title
plt.xlabel('Step (in millions)')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Steps')
plt.legend()

# Show the plot
plt.show()
'''