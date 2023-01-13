import pandas as pd
from generate_tables import metrics_mapping

bdf50 = pd.read_csv("outputs/score_csvs/parentbeam50.csv")

metrics_mapping( 'posthoc',bdf50,  'table')

bdf50.to_csv("outputs/score_csvs/parentbeam50phoc.csv")