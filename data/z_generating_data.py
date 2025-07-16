"""
    This is how I randomly generated star data (with help of ChatGPT).
    
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define ranges and categories for synthetic star data
num_rows = 100_000

# Star properties
radius = np.random.lognormal(mean=0, sigma=1, size=num_rows) * 0.7
temperature = np.random.normal(loc=5800, scale=4000, size=num_rows)
temperature = np.clip(temperature, 2000, 40000)  # clamp to realistic star temperatures

# Luminosity (log-normal based on temperature and radius)
luminosity = radius**2 * (temperature / 5778)**4

# Absolute Magnitude - inverse log of luminosity
absolute_magnitude = 4.83 - 2.5 * np.log10(luminosity)

# Star colors
colors = ['Red', 'Blue White', 'Yellow White', 'White', 'Orange', 'Blue']
star_color = np.random.choice(colors, size=num_rows)

# Spectral classes (O, B, A, F, G, K, M)
spectral_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
spectral_class = np.random.choice(spectral_classes, size=num_rows, p=[0.01, 0.05, 0.1, 0.2, 0.3, 0.25, 0.09])

# Star types (0 = Brown Dwarf, 1 = Red Dwarf, 2 = White Dwarf, 3 = Main Sequence, 4 = Supergiant, 5 = Hypergiant)
star_type = np.random.choice([0, 1, 2, 3, 4, 5], size=num_rows)

# Create DataFrame
star_data = pd.DataFrame({
    "Radius": radius,
    "Temperature": temperature,
    "Luminosity": luminosity,
    "Absolute_Magnitude": absolute_magnitude,
    "Star_Color": star_color,
    "Spectral_Class": spectral_class,
    "Star_Type": star_type
})

# Saving to '.csv' file
star_data.to_csv(r'..\data\star_data.csv', index=False)

# Preview the data
print(star_data)
