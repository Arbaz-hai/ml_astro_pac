to run the code
```
python run.py
```
## Data Overview
. Data is provided in CSV files with two types:
- Header files (metadata about astronomical objects)
- Light-curve data files (time series measurements)

## Header Files
Each row contains object properties with the following fields:

| Field | Description | Type |
|-------|-------------|------|
| `object_id` | Unique identifier | int32 |
| `ra` | Right ascension (longitude, degrees) | float32 |
| `decl` | Declination (latitude, degrees) | float32 |
| `gal_l` | Galactic longitude (degrees) | float32 |
| `gal_b` | Galactic latitude (degrees) | float32 |
| `ddf` | Flag for DDF survey area (1=DDF) | Boolean |
| `hostgal_specz` | Spectroscopic redshift (highly accurate) | float32 |
| `hostgal_photoz` | Photometric redshift (less accurate) | float32 |
| `hostgal_photoz_err` | Uncertainty on photometric redshift | float32 |
| `distmod` | Distance modulus calculated from photoz | float32 |
| `MWEBV` | Milky Way extinction factor | float32 |
| `target` | Astronomical source class (provided in training only) | int8 |

## Light-Curve Files
Each row represents an observation at a specific time and passband:

| Field | Description | Type |
|-------|-------------|------|
| `object_id` | Unique identifier matching header file | int32 |
| `mjd` | Modified Julian Date of observation | float64 |
| `passband` | LSST passband (u,g,r,i,z,y = 0,1,2,3,4,5) | int8 |
| `flux` | Measured brightness (MWEBV corrected) | float32 |
| `flux_err` | Uncertainty on flux measurement | float32 |
| `detected` | Significant detection flag (1=detected) | Boolean |

## Important Notes
- Training data: 1 light-curve file per object
- Data gaps exist between different passbands
- Milky Way objects have redshift of zero
- Flux can be negative due to statistical fluctuations
- Approximately 1% of data comes from DDF subset