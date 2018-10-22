#!/bin/bash
echo -ne "Pyan architecture: generating architecture.{dot,svg}\n"
/Users/wangyujie/Desktop/iProud/iCourse/US/294-Reinforcement_Learning/viz_code/pyan/pyan.py model_based_policy.py --defines --uses --colored --grouped --annotate --dot -V >model_based_policy.dot 2>model_based_policy.log
dot -Tsvg model_based_policy.dot >model_based_policy.svg
