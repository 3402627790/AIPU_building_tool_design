import pandas as pd


def convert_single_metric_to_pd(metric: dict):
    if len(metric) == 0:
        return []
    columns = []
    for k in metric.keys():
        k = eval(k)
        for t in k:
            columns.append(t[0])
        break
    columns.append("metric_score")
    pds = {}

    for k, v in metric.items():
        k = dict(eval(k))
        if len(v) == 0:
            continue
        for vk, vv in v.items():
            if vk not in pds.keys():
                pds[vk] = pd.DataFrame(columns=columns)
            data = k.copy()
            data["metric_score"] = vv
            pds[vk] = pds[vk].append(data, ignore_index=True)

    return pds
