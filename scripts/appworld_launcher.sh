#!/bin/bash

cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/

/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/.venv/bin/python \
    run_all.py \
    --split "${DATA_SPLIT}" \
    --model_name "${LLM}" \
    --tag baseline \
    --base-config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/configs/base_config.yaml \
    --co_config_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/configs/"${BENCHMARK}"/"${PROVIDER}"/"${SLURM_JOB_NAME}".yaml
