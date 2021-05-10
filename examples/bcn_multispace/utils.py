  import matplotlib.pyplot as plt

  def plotAll(model,figname):
    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()
    
    tot_people =model._blocks["density"]
    scheme = mapclassify.Quantiles(tot_people, k=5) 
 
    geoplot.choropleth(
      model._blocks, hue=tot_people, scheme=scheme,
      cmap='Oranges', figsize=(12, 8), ax=ax1
    )
    plt.savefig('density-'+figname)

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()

    ctx.plot_map(model._loc, ax=ax1)
    _c = ["red", "blue"]
    
    model._blocks.plot(ax=ax1,facecolor='none', edgecolor="black")  
    

    plt.tight_layout()

    # Plot agents
    self.grid._agdf.plot(ax=ax1)
    plt.savefig('agents-'+figname)
    
