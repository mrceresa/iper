class CityPlotter():
    def _plotAgents(self, outdir, figname):
        if self.space._gdf_is_dirty: self.space._create_gdf()

        fig = plt.figure(figsize=(20,20))
        ax1 = plt.gca()
        self.space._agdf["state"] = 0
        en = {"S":0,"R":1,"E":2,"A":3,"I":4,"H":5,"D":6}
        for a in self.space._agents.values():
          if not a.unique_id.startswith("Human"): continue
          self.space._agdf.at[a.unique_id,"state"] = en.get(a.machine.state) 

        self.space._agdf.plot(ax=ax1, column="state",markersize=1)
        ctx.add_basemap(ax=ax1,crs=self.space._crs)

        cmap = plt.cm.get_cmap('jet', len(en.keys())) 

        plt.legend([mpatches.Patch(color=cmap(b)) for b in range(len(en.keys()))],
           ['{}'.format(i) for i in ["S","R","E","A","I","H","D"]])

        #self._blocks.plot(ax=ax1, facecolor='none', edgecolor="black")
        
        ax1.set_axis_on()
        
        # Plot agents
        #self.space._agdf.plot(ax=ax1, legend=True)
        plt.savefig(os.path.join(outdir, 'agents-' + figname))       
        plt.close() 


    def _plotDensity(self,outdir, figname):
        fig = plt.figure(figsize=(15, 15))
        ax1 = plt.gca()

        tot_people = self._blocks["density"]
        scheme = mapclassify.Quantiles(tot_people, k=5)

        geoplot.choropleth(
            self._blocks, hue=tot_people, scheme=scheme,
            cmap='Oranges', figsize=(12, 8), ax=ax1
        )
        #plt.colorbar()
        plt.savefig(os.path.join(outdir, "density-" + figname))
        plt.close()
    
    def plotAll(self, outdir, figname):
        #self._plotDensity(outdir, figname)
        self._plotAgents(outdir, figname)
    
    def plot_results(self, outdir, title='stats', hosp_title='hosp_stats', R0_title='R0_stats'):
        """Plot cases per country"""
        self.l.info("PLOTTING RESULTS")
        self.l.info(self.Hospitalized_total)
        if isinstance(self.alarm_state['inf_threshold'], int) or self.alarm_state['inf_threshold'] == "2021-01-02":
            alarm_state = False
        else:
            alarm_state = True

        X = self.datacollector.get_table_dataframe("Model_DC_Table")

        X['Day'] = X['Day'].apply(pd.Timestamp)

        R0_df = X[['Day', 'R0', 'R0_Obs']]
        R0_df.to_csv(outdir + "/" + R0_title + str(self.general_run) + '.csv', index=False)  # get the csv

        ### R0 plot ###
        columns = ['R0', 'R0_Obs']  # , 'Mcontacts', 'Quarantined', 'Contacts']
        colors = ["Orange", "Green", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Rt values')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, R0_title))

        # Model stats plot
        X.drop(['R0', 'R0_Obs'], axis=1, inplace=True)
        X.to_csv(outdir + "/" + title + str(self.general_run) + '.csv', index=False)  # get the csv

        columns = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Hospitalized', 'Dead']
        colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Model stats')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, title))

        Y = self.hosp_collector.get_table_dataframe("Hosp_DC_Table")
        Y['Day'] = Y['Day'].apply(pd.Timestamp)
        Y.to_csv(outdir + "/" + hosp_title + str(self.general_run) + '.csv', index=False)  # get the csv

        # Hospital stats plot
        columns = ['Hosp-Susceptible', 'Hosp-Infected', 'Hosp-Recovered', 'Hosp-Hospitalized', 'Hosp-Dead']
        colors = ["Green", "Red", "Blue", "Gray", "Black"]

        Y.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Observed stats')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, hosp_title))