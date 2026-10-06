#!/usr/bin/env bash
S=code/business_entity_resolution/src
.venv/bin/python $S/test_pipeline.py --ce data/work/p11/ce_s2P.pt --ce-base e5 && \
.venv/bin/python $S/test_pipeline.py --recipe sub31_small --ce data/work/p11/ce_s2P.pt --ce-base e5 --unseen France
