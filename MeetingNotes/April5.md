# April 5th meeting notes

# discussed

    0. For the future, we need some tasks applications.
        0.1 predict future flows, like forcast #B.
        0.2 for paticular flow, given other attribute, predict like protocol.
        0.3 to say: how good the generated data is.

    1. 2 experiment: KL divergency and likelyhood. (log likelyhood)
        Strenth: PixelRNN maximum the log likelyhood.
    
    2. try co-lab for GPU issues.

    3. top-left to down-right, multiple each row / generate
        PixelCNN X assump dependency

    4. sample size of our block
        approach
            to make dependency assumption
            1st assumption: how far time can be trace back.
            
            temperally, manually.
    
    5. this affect our distribution factorization
        5.1 factorize with out doing assumption, like PixelRNN
        5.2 conditional proba distribution
        Mask let them to use 1NN, on same.

    6. Assume Mask are the same, which across time.
        parameter of condition consistance. 
        
        For each pixel, mask is same, which allows the dependency acutally exist.

        The #var that dependent on.

        Bayesian Network.

