import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Style, Fore

RC = {
    # "axes.facecolor": "#F8F8F8",
    # "figure.facecolor": "#F8F8F8",

    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

PALETTE = ['#302c36', '#037d97', '#91013E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']


class ColorStyle(object):
    red = Style.BRIGHT + Fore.RED
    blk = Style.BRIGHT + Fore.BLACK
    gld = Style.BRIGHT + Fore.YELLOW
    mgt = Style.BRIGHT + Fore.MAGENTA
    blu = Style.BRIGHT + Fore.BLUE
    res = Style.RESET_ALL


cS = ColorStyle()


def read_cfg(fp: str):
    """Reads configuration file"""
    with open(fp) as f:
        cfg = yaml.safe_load(f)
    return cfg


def plot_count(df: pd.DataFrame, col_list: list, title_name: str = 'Train') -> None:
    """Draws the pie and count plots for categorical variables.

    Args:
        df: train or test dataframes
        col_list: a list of the selected categorical variables.
        title_name: 'Train' or 'Test' (default 'Train')

    Returns:
        subplots of size (len(col_list), 2)
    """
    f, ax = plt.subplots(len(col_list), 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0)

    s1 = df[col_list].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1 / N

    outer_colors = ['#302c36']
    inner_colors = ['#EC0010']

    ax[0].pie(
        outer_sizes, colors=outer_colors,
        labels=s1.index.tolist(),
        startangle=0, frame=True, radius=1.3,
        explode=([0.05] * (N - 1) + [.3]),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    text_props = {
        'size': 13,
        'weight': 'bold',
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=0,
        autopct='%1.f%%', explode=([.1] * (N - 1) + [.3]),
        pctdistance=0.8, textprops=text_props
    )

    center_circle = plt.Circle((0, 0), .68, color='black',
                               fc='#FFF9ED', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = [0, 1]
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=[PALETTE[0], '#EC0010'],
        orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i + 0.1, str(v), color='black',
                   fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col_list, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()
