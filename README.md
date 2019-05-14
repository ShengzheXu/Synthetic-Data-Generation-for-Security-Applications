# Synthetic-Data-Generation-for-Security-Applications

## data preparation

use the link to download extracted raw data, processed data and the generated data. [[Dropbox](https://www.dropbox.com/s/uabnmy3b3b3ltna/data.zip?dl=0)]


If downloaded data in this way, you can go to step 4 directly.

## how to do a quick generation?

    Step1. Data Preparation
        Download the UGR16 data from the website, like april.week3.csv.uniqblacklistremoved. Then write it's path to the config.ini

        Delete the data under data/ folder for clean purpose.

    Step2. Analysis the occurence of the data and extract raw data of some users. The output of this step will be the data/raw_data/*.csv
        cd dataset/
        `python analyze_and_extract_ugr16.py -e`
    
    Step3. Clean and make up the raw data and generate the cleaned data. Then decide the bin size.
        `python process_ugr16_data.py -fw'

    Step4. Modify the config file to choose the baseline or later use the command line to override
        `python run_exp.py`

    Step5. Modify the config file to choose the validation model and Use KL to validate the generated data.
        `python validation.py'

## hyperparameters setting

    Refer to the config.ini file. Three categories settings are considered so far.
    
    1. [Default], making sure the basic functionality can work.
        userlist: ip users to be considered and learned.
        baseline: model choice.

    2. [GENERATE], describing generation ralevant issues.
        save_to_csv: whether output the generated data to the disk.
        gen_users: ip suers that to be generated as the sip.
        original_date, gen_daynumber: how many days to be generated and the start date of them.

    3. [VALIDATE], describing validation ralevant issues.
        test_set: file location that to be tested.
        raw_compare, gen_compare: 2 groups of experiments.

## design structure

    |~datasets/
    | |-UGR16.py: extract target data from the whole raw data
    | |-split_timestamp.py: craft data to the cleaned data
    |~models/
    | |-baselines.py: 3 classes, baseline1&2 class implementations and the father class which defines the interfaces
    | |-run_exp.py: encapsulated experiments entrance
    | |-validation.py: test modules
    |~output/
    | |+raw_data/: 10 csv files, which refer 10 users data of day1.
    | |+clean_data/: 10 csv files, which are supplemented by support info columns, like hour, delta time and #B-1
    | |+gen_data/: 2 csv files for baseline1&2. Each one contain 5 days generated data.
    | |-exp_record.txt: experiments parameters settings and process time

## baseline 1

    GMM - 7 cluster for the #B.

## baseline 2

    Baseline2 rule:
    argmax_B { P(B|T, B-1) = P(T|B) * P(B-1|B) * P(B) }

    Theoretically, the very first sample to be generated should be the marginal distribution P(B)

## data usage

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