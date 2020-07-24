====== Compartmental models 
In this folder we implemented several compartmental models and their fit to real data.

Results for each run are in the results folder, in a subfolder with the name of the model and today's date

* SIR

'python sir.py'

* SEIRD

'python seird.py fit --shift 10 -n 500000 --data data/dpc-covid19-ita-regioni.csv'


'python seird.py sim --r0_max 2.86 --r0_min 0.3 -k 3.45 --days 500 -n 500000 --rates rse=1.0,rei=0.2,rih=0.1,rir=0.06,rhr=0.1,rhd=0.1'


* SEIHRD

'python seihrd.py fit -n 500000'

* SEIHCRD 
