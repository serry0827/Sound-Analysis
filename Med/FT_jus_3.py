import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the fixed sample rate to 5000 Hz
sample_rate = 5000

# Read the CSV file
f = pd.read_csv("Med/Capstone Data(121~180).csv")  # Enter File Name

# Loop through the batches of runs (1-5, 6-10, etc.)
for batch in range(0, 60, 5):
    # Create a new figure with subplots for each batch of 5 runs
    fig, axs = plt.subplots(2, 5, figsize=(20, 12))  # 2 rows, 5 columns (side by side)
    
    # Initialize variables to store the common axis limits
    all_time_values = []
    all_intensity_values = []
    all_freq_values = []
    all_fft_values = []
    
    # Loop through the runs in the current batch (1-5, 6-10, etc.)
    for i, (ax_time, ax_freq) in zip(range(batch + 1, batch + 6), axs.T):
        # Get the time and sound intensity for the current run
        time_column = f.loc[:, f"Time (s) Run #{i}"]
        intensity_column = f.loc[:, f"Sound Intensity (V) Run #{i}"]
        
        # Remove NaN values
        data = pd.DataFrame({
            "Time": time_column,
            "Intensity": intensity_column
        }).dropna()
        
        # Collect time and intensity data for common scaling
        all_time_values.extend(data["Time"])
        all_intensity_values.extend(data["Intensity"])
        
        # Plot the full time-domain data for this run
        ax_time.plot(data["Time"], data["Intensity"], label=f"Run #{i}")
        
        # Apply the Fourier transform to the sound intensity data
        fft_values = np.fft.fft(data["Intensity"])
        
        # Get the frequencies corresponding to the FFT result
        freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
        
        # Remove the frequencies 0-50 Hz by setting the corresponding FFT values to 0
        fft_values[(freqs >= 0) & (freqs <= 100)] = 0
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft_values = np.abs(fft_values[:len(freqs)//2])  # Magnitude of the FFT
        
        # Collect frequency and FFT values for common scaling
        all_freq_values.extend(positive_freqs)
        all_fft_values.extend(positive_fft_values)
        
        # Plot the frequency-domain data (FFT result) for this run
        ax_freq.plot(positive_freqs, positive_fft_values, label=f"Run #{i}", color = "orange")
        
        # Customize the frequency-domain subplot for this run
        ax_freq.set_title(f"Run #{i} - Frequency Domain")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude of FFT")
        ax_freq.legend(loc="best")
        
        # Customize the time-domain subplot for this run
        ax_time.set_title(f"Run #{i} - Time Domain")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Sound Intensity (V)")
        ax_time.legend(loc="best")
    
    # Find the common axis limits for the time-domain plots
    time_min, time_max = min(all_time_values), max(all_time_values)
    intensity_min, intensity_max = min(all_intensity_values), max(all_intensity_values)
    
    # Find the common axis limits for the frequency-domain plots
    freq_min, freq_max = min(all_freq_values), max(all_freq_values)
    fft_min, fft_max = min(all_fft_values), max(all_fft_values)
    
    # Set the same axis limits for all subplots (time-domain)
    for ax_time in axs[0, :]:
        ax_time.set_xlim(time_min, time_max)
        ax_time.set_ylim(intensity_min, intensity_max)
    
    # Set the same axis limits for all subplots (frequency-domain)
    for ax_freq in axs[1, :]:
        ax_freq.set_xlim(freq_min, freq_max)
        ax_freq.set_ylim(fft_min, fft_max)
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Name the file dynamically based on the batch range, like "batch_1_to_5_plots.png"
    fig_filename = f"batch_{((batch // 5) * 5) + 121}_to_{((batch // 5) * 5) + 125}_plots.png"
    
    # Save the figure
    plt.savefig(fig_filename)
    print(f"Figure saved as {fig_filename}")
    
    # Show the figure (optional)
    #plt.show()

    # Close the figure to avoid memory issues when generating many figures
    plt.close(fig)