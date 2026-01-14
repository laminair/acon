#!/bin/bash

/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/.venv/bin/python \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/run_all.py \
    --split "${DATA_SPLIT}" \
    --model_name "${LLM}" \
    --tag baseline \
    --co_config_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/configs/acon/"${BENCHMARK}"/"${PROVIDER}"/"${SLURM_JOB_NAME}".yaml
