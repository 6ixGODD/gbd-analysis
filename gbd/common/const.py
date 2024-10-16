from __future__ import annotations

AGE_GROUPS = ['<5 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years', '30-34 years',
              '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', '60-64 years', '65-69 years',
              '70-74 years', '75-79 years', '80-84 years', '85-89 years', '90-94 years', '95+ years']

RAW_COLUMNS = ['measure_id', 'measure_name', 'location_id', 'location_name', 'sex_id',
               'sex_name', 'age_id', 'age_name', 'cause_id', 'cause_name', 'metric_id',
               'metric_name', 'year', 'val', 'upper', 'lower']

NUMBER_AGG_TEMPLATE = {'val': 'sum', 'upper': 'sum', 'lower': 'sum'}

RATE_AGG_TEMPLATE = {'val': 'mean', 'upper': 'mean', 'lower': 'mean'}

AGG_TEMPLATE = {
    'val_num':    'sum',
    'upper_num':  'sum',
    'lower_num':  'sum',
    'val_rate':   'mean',
    'upper_rate': 'mean',
    'lower_rate': 'mean',
    'population': 'sum',
}
