# Jan 17 Meeting Keynote

## target of the following week

try to implement a VAE model on the UGR’16 April Week3 data.

## discussion

1. Don't use the simulator like [3] paper. Develop a tool purely using data-to-data form and waive any expertise requirement.

2. model data dependency between the users.
    
    2.1 node/edge dependency. [user<->user], [each user's time series], [each user's dependent fields]

    2.2 graph connection

3. could generate different models for different tasks.

4. validation: could be the similarity to the original data. 
    
    4.1 using data from a live security setup environment. part of data to learn the classifier and another part to test
    4.2 another is using a deep learning model to learn the data. Then evaluate how good on the generated data and how good on original data.

5. the first step may be

    5.1 use statistics to model equivalent normal data
    5.2 model some abnormal sense in that data, thus, separate the two phases.
    5.3 combine GANs and VAE?

6. questions to answer.
    
    6.1 whether deep learning can generate the complexity of the data.


---

## preparation
[PPT](https://docs.google.com/presentation/d/1smBW6qmYNyvLvRZbiOUTC2-uz4CcKEMTJBD1EvNpA68/edit?usp=sharing)