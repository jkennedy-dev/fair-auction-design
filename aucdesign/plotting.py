import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def run_plotting() -> None:
    data = pd.read_csv("./results.csv")
    df1, df1_ctrl, df2, df2_ctrl = (pd.DataFrame() for _ in range(4))

    # Categorise trajectories
    for column in data:
        if column.find("type1_ctrl") != -1:
            df1_ctrl[column] = data[column]
        elif column.find("type1_") != -1:
            df1[column] = data[column]
        elif column.find("type2_ctrl") != -1:
            df2_ctrl[column] = data[column]
        elif column.find("type2_") != -1:
            df2[column] = data[column]

    # Convert df into a format that is useful for plotting
    def transform_df(df, label):
        df = df.reset_index().melt(id_vars="index", var_name="traj", value_name="value")
        df["group"] = label
        return df

    # Visualise data
    fig, (axes1, axes2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, sharey=True)
    sns.lineplot(
        data=pd.concat([transform_df(df1, "RAP"), transform_df(df1_ctrl, "Control")]),
        x="index",
        y="value",
        hue="group",
        linewidth=1,
        alpha=0.8,
        ax=axes1,
        estimator="mean",
        errorbar="ci",
    )
    sns.lineplot(
        data=pd.concat([transform_df(df2, "RAP"), transform_df(df2_ctrl, "Control")]),
        x="index",
        y="value",
        hue="group",
        linewidth=1,
        alpha=0.8,
        ax=axes2,
        estimator="mean",
        errorbar="ci",
    )

    # Configuration of legends
    axes1.legend(title=None, loc="lower right")
    axes2.legend(title=None)

    # Configuration for subplot 1
    axes1.set_title("Pr(+) = 0.4 & Pr(-) = 0.6")
    axes1.set_xlabel("Number of Auctions")
    axes1.set_ylabel("Inequality Score")
    axes1.set_xscale("log")
    axes1.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)

    # Configuration for subplot 2
    axes2.set_title("Pr(+) = 0.7 & Pr(-) = 0.3")
    axes2.set_xlabel("Number of Auctions")
    axes2.set_ylabel("Inequality Score")
    axes2.set_xscale("log")
    axes2.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig("visualised_data.png", dpi=300)

