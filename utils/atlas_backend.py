from foundations import load_parameters, log_params
from foundations import set_tensorboard_logdir
from foundations import save_artifact
from foundations import submit

metrics = {}


def log_metric(key, value):
    if isinstance(value, float):
        import numpy as np
        value = np.clip(value, 1e-10, 1e10)
        if np.isnan(value):
            value = "nan"
        else:
            value = float(value)
    elif isinstance(value, bool):
        value = 1 if value else 0
    elif isinstance(value, (str, int)):
        value = value
    else:
        raise TypeError("value must be float, int, bool, or str")

    metrics[key] = value
    import pickle
    with open('metric.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    save_artifact('metric.pkl', key='metric')

    from foundations import log_metric as flog_metric
    flog_metric(key, value)


def log(*args, **kwargs):
    print(*args, **kwargs)


log("using atlas framework")
name = 'atlas_backend'

"""
get hparams keys from atlas webpage
-------
all = ""
$($(".job-static-columns-container")[1]).find("h4").each(function(){
    all += $(this).text() + "|";
})
console.log(all)
-------
get hparams from atlas webpage
-------
all = ""
$($(".input-metric-column-container")[3]).find(".job-table-row").each(
    function(){
        ret = ""
        $(this).find(".job-cell").each(
        function(){
            ret += $($(this).find("p")[0]).text() + "|";
        });
    all += ret + ':\n';
});
console.log(all)
-------
get results from atlas webpage
-------
all = ""
$($(".input-metric-column-container")[5]).find(".job-table-row").each(
    function(){
        ret = ""
        $(this).find(".job-cell").each(
        function(){
            ret += $($(this).find("p")[0]).text() + "|";
        });
    all += ret + ':\n';
});
console.log(all)
-------
summarise the results
-------
param_keys = """"""
params = """"""
metrics = """"""

param_keys = param_keys.split("|")[:-1]
params = [p.split("|")[:-1] for p in params.replace("\r", "").replace("\n", "").split(":")[:-1]]
metrics = [p for p in metrics.replace("\r", "").replace("\n", "").split(":")[:-1]]
assert len(params) == len(metrics)
assert len(params[0]) == len(param_keys)

selected = []
for i in range(len(params[0])):
    for p in params:
        if p[i] != params[0][i]:
            selected += [i]
            break
param_keys = [v for i,v in enumerate(param_keys) if i in selected]
params = [[v for i,v in enumerate(p) if i in selected] for p in params]

print("|".join(param_keys))
print("|".join(["---"] * len(param_keys)))
for p, m in zip(params, metrics):
    print("|".join(p), "|", m)
------- 
"""