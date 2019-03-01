# Feb 22 Meeting Keynote

## Meeting with Prof. Randy Marchany

## Validation for baseline1&2

    K-l divergency and NMI

## Next Step

1. about Var Bayesian/ Var latent, existed bayesian network architecture.

    Vae hidden latent/observable, which is a predicting parameter latent. Encoder-decoder for latent var - raw data.

2. start with some such architecture.
    
    They assume id rows are independent. To double check this, check papers about temporal relationship dependent.

3. idea: combining RNN with Bayesian network

    use CNN, RNN to estimate distribution

    then provide conditional probability
    and maximize the likelyhood of data

    that's where var bayesian comes.

4. Currently many similar works are for Imgs/NLP

        VAE <-> GAN
        VAE: combing Bayesian and Deep learning
        GAN: don't max likelyhood

        have been applied