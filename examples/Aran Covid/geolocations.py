import requests, urllib, json, pandas, csv

from bs4 import BeautifulSoup
import urllib3

http = urllib3.PoolManager()

# Second DB
#link = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=3cde43cd-d53a-4048-be38-4e0f8a384eea&fields=EQUIPAMENT,LATITUD,LONGITUD,3ER_NIVELL,2N_NIVELL,1ER_NIVELL"

# First DB
link = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search_sql?sql=SELECT DISTINCT(name),geo_epgs_4326_x,geo_epgs_4326_y,secondary_filters_name from \"9e135848-eb0a-4bc5-8e60-de558213b3ed\" WHERE secondary_filters_name IN ('Hospitals i clíniques', 'CAPs', 'Centres urgències (CUAPs)' )"

r = http.request('GET', link)
data = json.loads(r.data.decode('utf-8'))

db = pandas.read_json(json.dumps(data['result']['records']))
#db.to_csv('pandas.csv')
print(db.head(10))
print(db.shape)
print(db.secondary_filters_name.unique())

