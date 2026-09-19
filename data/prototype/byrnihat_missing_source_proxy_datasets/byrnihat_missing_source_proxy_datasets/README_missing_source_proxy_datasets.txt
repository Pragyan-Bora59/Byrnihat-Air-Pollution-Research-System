Byrnihat Missing-Source Proxy Datasets
Generated for: Byrnihat air pollution + meteorology project
Date range: 2026-01-01 00:00:00 to 2026-06-28 23:00:00, hourly
Important status: SYNTHETIC PROXY DATA, not official AQICN/NASA/IMD/CPCB observations.

Purpose:
These files fill the current gap where live collection is still incomplete:
1. Pollution data for Byrnihat, Guwahati, and Shillong.
2. Meteorological data around Byrnihat.
3. A merged ML-ready dataset for testing ML, graphs, dashboard, and simulation integration.

Do not use these files for final scientific claims, papers, or government-facing conclusions.
Use them only for pipeline development, model testing, graph testing, and dashboard integration.
When real AQICN/CPCB/NASA/IMD data becomes available, replace these proxy files.

Recommended project placement:
data/prototype_missing_sources/
├── aqicn_observation_schema/
│   ├── prototype_byrnihat_aqicn_observations.csv
│   ├── prototype_guwahati_aqicn_observations.csv
│   ├── prototype_shillong_aqicn_observations.csv
│   └── prototype_all_three_city_pollution_observations.csv
├── meteorology_schema/
│   ├── prototype_byrnihat_region_meteorological_observations.csv
│   └── prototype_nasa_power_hourly_tracker_byrnihat_region.csv
└── ml_ready/
    ├── prototype_ml1_pollution_meteorology_dataset.csv
    ├── prototype_ml_train.csv
    └── prototype_ml_test.csv

Pollution schema:
The city pollution files are shaped to be close to the new processed AQICN observation schema:
first_collection_time_utc, target_city, station_code, station_name, station_latitude,
station_longitude, aqicn_idx, api_time, aqi, pm25, pm10, no2, so2, co, o3,
nh3, no, nox, temperature, humidity, pressure, wind.

Meteorology schema:
The meteorology file includes hourly weather proxy data around Byrnihat:
temperature, humidity, pressure, rainfall, wind_speed, wind_direction,
wind_to_direction, wind_u, wind_v, boundary_layer_height_proxy_m,
ventilation_index_proxy.

ML-ready target columns:
pm25_next_hour_target
pm10_next_hour_target
aqi_next_hour_target
pm25_change_next_hour_target
pm10_change_next_hour_target
transport_direction_deg_target
transport_direction_sector_target

Synthetic generation logic:
The values follow simplified, physics-inspired relationships:
- higher emissions increase PM2.5/PM10/NO2/SO2/CO
- rainfall lowers particulate matter through washout
- stronger wind increases dispersion
- humidity can increase particle growth
- diurnal traffic peaks influence NO2/CO/PM
- wind vectors are derived from wind speed and wind-from direction

This dataset is intentionally transparent and labelled using:
data_status = synthetic_proxy_missing_source
intended_use = ml_training_graph_testing_pipeline_debugging_only
