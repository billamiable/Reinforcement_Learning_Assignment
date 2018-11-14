#!/bin/bash
echo -ne "Pyan architecture: generating architecture.{dot,svg}\n"
/Users/wangyujie/Desktop/iProud/iCourse/US/294-Reinforcement_Learning/viz_code/pyan/pyan.py train_policy.py --defines --uses --colored --grouped --annotate --dot -V >train_policy.dot 2>train_policy.log
dot -Tsvg train_policy.dot >train_policy.svg
