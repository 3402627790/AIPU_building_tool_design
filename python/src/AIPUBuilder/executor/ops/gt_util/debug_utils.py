import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def show_plt(graph_debug_info):
    index_name = "dumpIR layer_id"
    index_data = graph_debug_info["ids"]
    model_cos = "cos"
    cos_data = graph_debug_info[model_cos]
    model_mse = "mse"
    mse_data = graph_debug_info[model_mse]

    if not all([index_data, cos_data, mse_data]):
        return

    def no_scientific_formatter(x, pos):
        return f'{x:.6f}'

    def show_maxmin_plot(x, y, show_min = True):
        # all_indices_sorted = np.argsort(y)
        # k = min(3, len(y))
        # max_indices = all_indices_sorted[-k:]
        # min_indices = all_indices_sorted[:k]
        # for i in range(k):
        #     xx = x[max_indices[i]]
        #     yy = y[max_indices[i]]
        #     if show_min:
        #         xx = x[min_indices[i]]
        #         yy = y[min_indices[i]]
        #     plt.annotate(
        #         f"{index_name}: {xx}\n{yy}",
        #         (xx, yy),
        #         textcoords="offset points",
        #         xytext=(0, 10),
        #         ha="left",
        #         color="red",
        #     )
        #     plt.plot(
        #         xx, yy, "o", markersize=5, markerfacecolor="red", markeredgecolor="red"
        #     )
        index_y_min = np.array(y).argmin()
        index_y_max = np.array(y).argmax()
        xx = x[index_y_max]
        yy = y[index_y_max]
        if show_min:
            xx = x[index_y_min]
            yy = y[index_y_min]
        plt.annotate(
            f"{index_name}: {xx}\n{yy}",
            (xx, yy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            color="red",
        )
        plt.plot(
            xx, yy, "o", markersize=5, markerfacecolor="red", markeredgecolor="red"
        )

    def create_scatter(x, y, x_label, y_label, formatter, show_min = True):
        plt.scatter(x, y, alpha=0.5)#, label=y_label)
        show_maxmin_plot(x, y, show_min)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xticks([]) #(rotation=45, ha='right')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend()

    def create_lineplot(x, y, x_label, y_label, formatter, show_min = True):
        plt.plot(x, y)
        show_maxmin_plot(x, y, show_min)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xticks([]) #(rotation=45, ha='right')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend()#(labels=[y_label])

    def create_histplot(data, data_label, formatter, show_min = True):
        index_y_min = np.array(data).argmin()
        index_y_max = np.array(data).argmax()

        # sns.histplot(cos_dist, kde=True, alpha=0.5, label=model_cos)
        plt.hist(data, alpha=0.5, label=data_label)
        line_maxmin = data[index_y_max]
        line_label = "Max"
        if show_min:
            line_maxmin = data[index_y_min]
            line_label = "Min"
        plt.axvline(line_maxmin, color="red", linestyle="--", label=line_label)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xlabel(data_label)
        plt.ylabel("count")
        plt.legend()

    formatter = FuncFormatter(no_scientific_formatter)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    create_scatter(index_data, cos_data, index_name, model_cos, formatter, True)
    plt.subplot(2, 3, 4)
    create_scatter(index_data, mse_data, index_name, model_mse, formatter, False)
    plt.subplot(2, 3, 2)
    create_lineplot(index_data, cos_data, index_name, model_cos, formatter, True)
    plt.subplot(2, 3, 5)
    create_lineplot(index_data, mse_data, index_name, model_mse, formatter, False)
    plt.subplot(2, 3, 3)
    create_histplot(cos_data, model_cos, formatter, True)
    plt.subplot(2, 3, 6)
    create_histplot(mse_data, model_mse, formatter, False)

    plt.tight_layout()
    # plt.show()
    plt.savefig("compare.png", bbox_inches="tight")