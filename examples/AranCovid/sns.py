import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import argparse


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # GRAPH WITH MODEL STATS
    colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]
    my_pal = {"Susceptible": "Green", "Exposed": "Yellow", "Infected": "Red", "Recovered": "Blue",
              "Hospitalized": "Gray", "Dead": "Black"}
    get_stats_from_files("stats", title='Model stats', colors=colors, my_pal=my_pal, output_dir=args.output_dir,
                         input_dir=args.input_dir, file_name="GENERAL_STATS")

    # GRAPH WITH OBSERVED_STATS
    colors = ["Green", "Red", "Blue", "Gray", "Black"]
    my_pal = {"Hosp-Susceptible": "Green", "Hosp-Infected": "Red", "Hosp-Recovered": "Blue",
              "Hosp-Hospitalized": "Gray", "Hosp-Dead": "Black"}
    get_stats_from_files("hosp_stats", title='Observed stats', colors=colors, my_pal=my_pal, output_dir=args.output_dir,
                         input_dir=args.input_dir, file_name="GENERAL_HOSP_STATS")

    # GRAPH WITH R0 STATS
    colors = ["Orange", "Green"]
    my_pal = {"R0": "Orange", "R0_Obs": "Green"}
    get_stats_from_files('R0_stats', title="R0 stats", colors=colors, my_pal=my_pal, output_dir=args.output_dir, input_dir=args.input_dir, file_name="General_R0_Stats")


def get_stats_from_files(start_with, title, colors, my_pal, output_dir, input_dir, file_name):
    stats_files = []
    for file in os.listdir(sys.path[0] + "/" + input_dir):
        if file.startswith(start_with):
            stats_files.append(file)

    li = []

    for filename in stats_files:
        df = pd.read_csv(input_dir + "/" + filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['Day'] = frame['Day'].apply(pd.Timestamp)

    columns = frame.columns[1:]

    for col in range(0, len(columns)):
        sns.lineplot(data=frame, x="Day", y=columns[col], color=colors[col], legend='brief')

    plt.ylabel('Values')
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.gcf().legend(labels=columns)
    plt.savefig(os.path.join(output_dir, 'Lineplot ' + file_name))
    plt.cla()
    plt.clf()

    if 'R0' not in columns:
        frame = frame.sort_values(by="Day")

        frame = frame[frame['Day'] == frame.at[df.index[-1], 'Day'].strftime('%Y-%m-%d')]  # get the last  day values
        frame.drop(['Day'], axis=1, inplace=True)

    else:
        frame = pd.concat(li, axis=0, ignore_index=True)
        frame['Day'] = frame['Day'].apply(pd.Timestamp)
        frame = frame.groupby(['Day']).mean()  # get the mean value grouped by days



    g = sns.boxplot(x="variable", y="value", data=pd.melt(frame), palette=my_pal)
    g.set(xlabel=None)

    plt.ylabel('Values')
    plt.title(title + ' boxplot')
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(output_dir, "Boxplot " + file_name))
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="OUTPUT GENERAL GRAPHS", help="Output dir")
    parser.add_argument('-i', '--input_dir', type=str, default="", help="Input dir")

    parser.set_defaults(func=main)

    args = parser.parse_args()
    model = args.func(args)
