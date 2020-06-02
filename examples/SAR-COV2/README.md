# SARS-COV2

## Summary

This model is based on current investigations in COVID spread.

## How to Run

To run the model interactively, run: 
```
    $ python run.py
```

To run the 0-dimensional models on real data (In this case, Trento italian province): 
```
    $ cd 0dim
    $ python seird2.py fit --shift 60 
```

This will result in a prediction for the parameters. To see the model with the fitted parameters:

 ```
    $ python seird2.py sim --r0_max 2.86 --r0_min 0.3 -k 3.45 --lock 99 --gamma 0.107 --delta 0.104 --rho 0.1 --days 500 -n 5000
```

Remember to install all iper code before as explained in the main page. 
