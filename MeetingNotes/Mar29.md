# March 29th meeting notes

# discussed

    Our baseline 1. Pi P_j(X_ij), where j is the column and P_j <- GMM
    
    P(X), where X is the whole matrix.

    n*m, where n is the n samples and m attributes.

    P(X) = P(X_11, X_12,..., X_n1, X_nm)

    for baseline 1, joint probability = Pi (X_ij)
    likelyhood of the data show better model have higher likelyhood (often use log likelyhood)

    ===========


    We start with joint distribution; baseline 1&2 go with simple assumption.
    
    When training it, we can calculate the likelyhood.

    Show the effect of the subsample

    same user, different day: continue;

    different user, start from different beginning.

# taks

    1. how we model distributions using the neural.

    2. compare KL divergence of gen_data, real data; and within real data.
    
    we calc average between each kl, and cal variance.

    3. coding part save the model, read the model from some script.

    4. gen data 10 users -> 1 day

    5. set what date to start to be a hyperparameter.