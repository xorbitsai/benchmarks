# Census 

This is a simple ETL and ML based on US census data.

### Data

Download the data with the following [link](https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz). And unzip the file with `gunzip` or other tools.

### Run

Run the benchmark, change `census_pandas.py` with other framework like `census_xorbits.py`.

```bash
python census_pandas.py --path /path/to/csv
```