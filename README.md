# Synthetic-Data-Generation-for-Security-Applications

|~data/
| |-UGR16.py: extract target data from the whole raw data
| |-split_timestamp.py: craft data to the cleaned data
|~algorithm/
| |-baselines.py: 3 classes, baseline1&2 class implementations and the father class which defines the interfaces
| |-run_exp.py: encapsulated experiments entrance
| |-validation.py: test modules
|~output/
| |+raw_data/: 10 csv files, which refer 10 users data of day1.
| |+clean_data/: 10 csv files, which are supplemented by support info columns, like hour, delta time and #B-1
| |+gen_data/: should be 22 csv files, which are generated data. 2×10+2 for (baseline1,2)×[(10users)+(avg during process)
| |-exp_record.txt: experiments parameters settings and process time

argmax_B { P(B|T, B-1) = P(T|B) * P(B-1|B) * P(B) }

42.219.153.7,baseline1 ==> time:104.46666666666667mins
42.219.153.89,baseline1 ==> time:96.9mins
42.219.155.56,baseline1 ==> time:98.23333333333333mins
42.219.155.26,baseline1 ==> time:68.6mins
42.219.159.194,baseline1 ==> time:52.75mins
42.219.152.249,baseline1 ==> time:48.43333333333333mins
42.219.159.82,baseline1 ==> time:49.666666666666664mins
42.219.159.92,baseline1 ==> time:48.61666666666667mins
42.219.159.94,baseline1 ==> time:49.56666666666667mins
42.219.158.226,baseline1 ==> time:27.433333333333334mins

42.219.153.7,baseline2 ==> time:164.36666666666667mins
42.219.153.89,baseline2 ==> time:155.83333333333334mins
42.219.155.56,baseline2 ==> time:77.23333333333333mins
42.219.155.26,baseline2 ==> time:64.75mins
42.219.159.194,baseline2 ==> time:65.2mins
42.219.158.226,baseline2 ==> time:163.03333333333333mins

ten_ip,baseline1 ==> time:342.28333333333336mins
ten_ips,baseline2 ==> time:657.7mins

    # 1366940, '42.219.153.7'      
    # 1342589, '42.219.153.89'     
    # 1210025, '42.219.155.56'
    # 1175793, '42.219.155.26'
    # 1081080, '42.219.159.194'
    # 1046866, '42.219.152.249'
    # 944492, '42.219.159.82'
    # 888781, '42.219.159.92'
    # 771349, '42.219.159.94'
    # 637608, '42.219.158.226' 