window_size=3

today=$(date +%Y-%m-%d)
# today=$(date -d "1 day ago" +%Y-%m-%d)

start_time=$(date +%s)

# update stock
while [ ! -f "stockdata/2023_by_date/${today}.csv" ] 
do
    python update_stockdata_by_date.py $today
done
pip install akshare==1.3.4

# update concept
rm stockdata/2023_by_date/concept_hist.csv
while [ ! -f "stockdata/concept_hist.csv" ] 
do
    python update_concept_hist.py
done
pip install akshare==1.9.55

# train model
while [ ! -f "submit/${today}_window${window_size}.csv" ] 
do
    python concept_train.py $window_size $today
done


end_time=$(date +%s)
runtime=$((end_time - start_time))
