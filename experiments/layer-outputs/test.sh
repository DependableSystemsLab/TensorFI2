#!/bin/sh

for i in $(seq 1 1 50)
do
      
      echo "Running experiment $i time"
      rm confFiles/sample.yaml
      cp bulk_experiments_default.conf ./confFiles/sample.yaml
      echo "Amount: ${i}" >> ./confFiles/sample.yaml     

      python3 -W ignore cnn-mnist.py ./confFiles/sample.yaml ./result/ 500 2

      cp ./result/res.csv ./result/"${i}".csv
      rm ./result/res.csv 
done
