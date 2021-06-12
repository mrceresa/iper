import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import argparse
from scipy import stats
import numpy as np


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # GRAPH WITH MODEL STATS
    colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]
    my_pal = {"Susceptible": "Green", "Exposed": "Yellow", "Infected": "Red", "Recovered": "Blue",
              "Hospitalized": "Gray", "Dead": "Black"}
    get_stats_from_files("stats", title='Model stats', colors=colors, my_pal=my_pal, output_dir=args.output_dir,
                         input_control_dir=args.input_control_dir, input_test_dir=args.input_test_dir,
                         file_name="General_STATS")

    # GRAPH WITH OBSERVED_STATS
    colors = ["Green", "Red", "Blue", "Gray", "Black"]
    my_pal = {"Hosp-Susceptible": "Green", "Hosp-Infected": "Red", "Hosp-Recovered": "Blue",
              "Hosp-Hospitalized": "Gray", "Hosp-Dead": "Black"}
    get_stats_from_files("hosp_stats", title='Observed stats', colors=colors, my_pal=my_pal, output_dir=args.output_dir,
                         input_control_dir=args.input_control_dir, input_test_dir=args.input_test_dir, file_name="General_HOSP_STATS")

    # GRAPH WITH R0 STATS
    colors = ["Orange", "Green"]
    my_pal = {"R0": "Orange", "R0_Obs": "Green"}
    get_stats_from_files('R0_stats', title="R0 stats", colors=colors, my_pal=my_pal, output_dir=args.output_dir, input_control_dir=args.input_control_dir, input_test_dir=args.input_test_dir, file_name="General_R0_Stats")


def mean_or_last_row(df, columns):
    if 'R0' in columns:
        df = df.groupby(['Day']).mean()  # get the mean value grouped by days

    else:
        df = df.sort_values(by="Day")
        df = df[df['Day'] == df.at[df.index[-1], 'Day'].strftime(
            '%Y-%m-%d')]  # get the last  day values
        df.drop(['Day'], axis=1, inplace=True)

    return df


def get_stats_from_files(start_with, title, colors, my_pal, output_dir, input_control_dir, input_test_dir, file_name):

    # CONTROL SET
    control_df = read_files(input_control_dir, start_with)
    lineplot(control_df, colors, title, input_control_dir, output_dir, file_name)
    columns = control_df.columns
    control_df = mean_or_last_row(control_df, columns)

    # TEST SET
    test_df = read_files(input_test_dir, start_with)
    lineplot(test_df, colors, title, input_test_dir, output_dir, file_name)
    test_df = mean_or_last_row(test_df, columns)

    """g = sns.boxplot(x="variable", y="value", data=pd.melt(frame), palette=my_pal)
    g.set(xlabel=None)

    plt.ylabel('Values')
    plt.title(title + ' boxplot')
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(output_dir, "Boxplot " + file_name))
    plt.cla()
    plt.clf()"""

    # * Statistical tests for differences in the features across groups
    all_t = list()
    all_p = list()

    columns = control_df.columns
    for col in columns:
        g1 = control_df[col].values
        g2 = test_df[col].values
        t, p = stats.ttest_ind(g1, g2)
        all_t.append(round(t,3))
        all_p.append(round(p, 3))

    #print(all_t, all_p)
    # print(np.count_nonzero(np.array(columns)[np.array(all_p) < 0.05])) # see that there is a statistically significant difference in all features

    # renaming so that class 0 will appear as setosa and class 1 as versicolor
    control_df['Group'] = 'Control'
    test_df['Group'] = 'Test'

    df = pd.concat([control_df, test_df])
    df_long = pd.melt(df, 'Group', var_name='Feature', value_name='Value')  # this is needed for the boxplots later on

    # Boxplots
    if 'R0' in columns: fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=100)
    else: fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=100)
    axes = axes.flatten()

    for idx, feature in enumerate(columns):
        ax = sns.boxplot(x="Feature", hue="Group", y="Value", data=df_long[df_long.Feature == feature], linewidth=2,
                         showmeans=True,
                         meanprops={"marker": "*", "markerfacecolor": "white", "markeredgecolor": "black"},
                         ax=axes[idx])

        # * tick params
        #axes[idx].set_title(feature)
        axes[idx].set_xticklabels([str(feature) ], rotation=0)
        axes[idx].set(xlabel=None)
        axes[idx].set(ylabel=None)
        axes[idx].grid(alpha=0.5)
        axes[idx].legend(loc="center right", prop = {"size": 11})

        # * set edge color = black
        for b in range(len(ax.artists)):
            ax.artists[b].set_edgecolor("black")
            ax.artists[b].set_alpha(0.8)

        # * statistical tests
        x1, x2 = -0.20, 0.20
        y, h, col = df_long[df_long.Feature == feature]["Value"].max() + 1, 2, 'k'
        axes[idx].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        axes[idx].text((x1 + x2) * .5, y + h, "t: " + str(all_t[idx]) + " p: " + str(all_p[idx]), ha ='center', va ='bottom', color = col)
        fig.suptitle("Significant feature differences between control and test groups", size=14, y=0.93)

    plt.tight_layout()
    plt.subplots_adjust(top=.88)
    plt.savefig(os.path.join(output_dir, 'Boxplot ' + file_name))
    plt.close()
    #plt.show()


def read_files(dir, start_with):
    stats_files = []
    for file in os.listdir(sys.path[0] + "/" + dir):
        if file.startswith(start_with):
            stats_files.append(file)

    li = []

    for filename in stats_files:
        df = pd.read_csv(dir + "/" + filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['Day'] = frame['Day'].apply(pd.Timestamp)

    return frame

def lineplot(frame, colors, title, input_dir, output_dir, file_name):
    columns = frame.columns[1:]

    for col in range(0, len(columns)):
        sns.lineplot(data=frame, x="Day", y=columns[col], color=colors[col], legend='brief')

    plt.ylabel('Values')
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.gcf().legend(labels=columns)
    plt.savefig(os.path.join(output_dir, 'Lineplot ' + input_dir + " " + file_name))
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="OUTPUT GRAPHS", help="Output dir")
    parser.add_argument('-c', '--input_control_dir', type=str, default="control", help="Input control dir")
    parser.add_argument('-t', '--input_test_dir', type=str, default="test", help="Input test dir")

    parser.set_defaults(func=main)

    args = parser.parse_args()
    model = args.func(args)
