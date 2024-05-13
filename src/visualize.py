import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file to check its content
data = pd.read_csv('out.csv')


# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(data['byte']/1e6, data['nJ']/1e6, marker='o', linestyle='')
plt.title('Energy Consumption vs. SRAM Size', fontsize=18)
plt.xlabel('SRAM Size (MB)', fontsize=16)
plt.ylabel('Energy Consumption (mJ)', fontsize=16)
plt.grid(True)
plt.savefig('plot.png')

plt.figure(figsize=(12, 8))
plt.plot(data['byte']/1e6, data['dram_acc'], marker='o', linestyle='')
plt.title('DRAM Access vs. SRAM Size', fontsize=18)
plt.xlabel('SRAM Size (MB)', fontsize=16)
plt.ylabel('DRAM Access (#)', fontsize=16)
plt.grid(True)
plt.savefig('plot1.png')