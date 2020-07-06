#!/usr/bin/env bash

nohup python -u use_bert.py --cuda 3,4,5 > log 2>&1 &