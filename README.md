# NIPA - Climate Index Analysis

A Python-based framework for analyzing the relationship between global climate indices and local climate variables. This project investigates how large-scale atmospheric and oceanic patterns influence regional precipitation and temperature anomalies.

## Overview

NIPA (readaptation of the original NIPA module) analyzes correlations between:
- **Climate Indices**: NAO (North Atlantic Oscillation), EA (East Atlantic), SCA (Scandinavian Pattern), ENSO-MEI (El Niño Southern Oscillation - Multivariate ENSO Index)
- **Atmospheric/Oceanic Fields**: Sea Surface Temperature (SST), Mean Sea Level Pressure (MSLP), 500mb Geopotential Height (Z500)
- **Local Climate Variables**: Precipitation and temperature anomalies at regional scales

The framework performs multi-lag analysis with configurable aggregation levels and time periods to identify statistically significant relationships.

## Project Structure

```
newNIPA/
├── Run_nipa.py                 # Main execution script (user-configurable analysis)
├── simpleNIPA.py               # Core NIPA analysis classes and methods
├── build_table.py              # Generate summary tables from Pearson correlations
├── climdiv_data.py             # Climate divisional data loading and processing
├── atmos_ocean_data.py         # Atmospheric and oceanic data loading
├── utils.py                    # Utility functions and helpers
├── DATA/                       # Raw input data files
│   ├── EA.txt, NAO.txt, SCA.txt, ENSO-mei.txt    # Climate indices
│   └── t2m_netherlands.txt, tp_netherlands.txt   # Local observations
├── maps/                       # Generated correlation maps (by climate pattern type)
├── output/                     # Analysis results
│   └── [Index]_[Field]_[Lag]/  # Results organized by index, field, and lag
│       ├── *_dataset.csv       # Full analysis dataset
│       ├── *_timeseries.csv    # Aggregated time series
│       └── *_pc1SST.csv        # SST principal component data
├── pearson_tables/             # Pearson correlation summary tables
└── specfiles/                  # Conda environment specifications by OS
```

## Features

- **Multi-lag correlation analysis**: Investigate delayed responses with configurable lag periods
- **Aggregation flexibility**: Analyze at multiple temporal scales (1, 2, 3-month aggregations)
- **Phase-based analysis**: Stratify analysis by positive/negative phases of climate indices
- **Statistical outputs**: Generate correlation grids, time series, and principal component analyses
- **Visualization**: Create correlation maps for different climate patterns and fields
- **Cross-platform support**: Pre-configured environments for Windows, macOS, and Linux/Ubuntu

## Installation

### Prerequisites
- Python 3.7+
- Conda package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd newNIPA
```

2. **Create environment from specfile** (choose your OS):

**Windows**:
```bash
conda create --name newnipa --file specfiles/spec-fileWindows.txt
conda activate newnipa
```

**macOS**:
```bash
conda create --name newnipa --file specfiles/spec-fileMacOS.txt
conda activate newnipa
```

**Ubuntu/Linux**:
```bash
conda create --name newnipa --file specfiles/spec-fileUbuntu.txt
conda activate newnipa
```

## Usage

### Running an Analysis

Edit the configuration section in [Run_nipa.py](Run_nipa.py#L18) to specify your analysis parameters:

```python
#### USER INPUT ####
local_datas = ['tp_netherlands_cumul']      # Local climate variables to analyze
aggrs = [1, 2, 3]                            # Aggregation levels (months)
months_complete = [i+1 for i in range(12)]  # Months to include (1-12)
indices = ['SCA','EA','ENSO-mei','NAO']      # Climate indices to use
global_datas = ['SST','MSLP','Z500']         # Global fields to correlate
n_comp = 1                                   # Number of principal components
```

Then execute:
```bash
python Run_nipa.py
```

### Key Analysis Parameters

- **Aggregation levels**: Temporal aggregation of climate indices (1, 2, or 3 months)
- **Lags**: Delay between global fields and local climate variable (configurable)
- **Phases**: Positive/negative phases of climate indices for stratified analysis
- **Months**: Specific months or seasons to focus on

## Data

### Input Data (DATA/ folder)
- **Climate indices**: Daily/monthly values for NAO, EA, SCA, ENSO-MEI
- **Local observations**: Time series of temperature and precipitation anomalies
- **Format**: Text files with time-indexed numerical data

### Output Data (output/ folder)
Results are organized as: `[Index]_[Global_Field]_[Lag]_number/`

**Output files per configuration**:
- `*_dataset.csv`: Complete dataset with aligned variables
- `*_timeseries.csv`: Aggregated time series for analysis
- `*_pc1SST.csv`: First principal component of SST correlations

### Generated Maps (maps/ folder)
Correlation maps visualizing spatial patterns for each combination of:
- Climate index (EA, ENSO-mei, NAO, SCA)
- Global field (MSLP, SST, Z500)
- Local variable (t2m, tp)

## Main Modules

- **simpleNIPA.py**: Core `NIPAphase` class for phase-based correlation analysis
- **atmos_ocean_data.py**: Data loading functions for global atmospheric/oceanic fields
- **climdiv_data.py**: Regional climate divisional data processing
- **build_table.py**: Aggregates results into summary Pearson correlation tables
- **utils.py**: Helper functions for data manipulation and visualization

## Configuration Files

Environment specifications for reproducibility are stored in [specfiles/](specfiles/):
- `spec-fileWindows.txt`: Windows dependencies
- `spec-fileMacOS.txt`: macOS dependencies  
- `spec-fileUbuntu.txt`: Ubuntu/Linux dependencies

## Output and Results

Analysis results are saved in structured directories with:
- **Correlation grids**: Spatial correlation maps
- **Time series**: Aggregated and lagged time series
- **Statistical tables**: Pearson correlation coefficients and significance
- **Principal components**: PCA-derived predictors from SST fields

## Requirements

Core dependencies include:
- pandas, numpy: Data manipulation and numerical computing
- xarray: Multi-dimensional array analysis
- matplotlib: Visualization
- scipy, scikit-learn: Statistical analysis

See `specfiles/` for complete dependency lists.
