# Mar 8 Meeting Keynote

Questions:

why NN can be used to express complex transform. Use NN to model the complex condition distribution.

## Pixel RNN/CNN

    Category: generative Model -> explicit density -> tractable density -> [Pixel RNN/CNN]

    1. fully visible belief network
        p(x) = pi(1~n) [p(xi | x1,..., xi-1)]
        
        Training: then maximize likelyhood of training data.

    2. how to model this?
        2.1 Complex distribution over pixel values => express using a neural network!

        2.2 will need to define ordering of "previous pixels"

        2.3 Both Pixel RNN/CNN generate image pixels starting from corner.

    3. Pixel RNN (van der Oord et al. 2016)
        3.1 Dependency on previous pixels modeled using an RNN (LSTM)
        3.2 Drawback: sequential generation is slow.

    4. Pixel CNN (van der Oorder et al. 2016)
        4.1 Dependency on previous pixels now modeled using a CNN over context region.
        4.2 Still start from the corner and use CNN with the softmax module (0-255) to generate the new pixel one by one.
        4.3 The 'training label' is the training data itself so we don't want any external labels.
        4.4 We're outputing a distribution over pixel values at each location. We want to maximize the likelihood of our input training data being generated. Use the training data to create the loss.
    
    5. Comparason Pixel CNN/RNN
        5.1 Training for PCNN is faster than PRNN. Generation time is slow on both model.
        5.2 PixelRNN/CNN allow you to explicitly compute likelihood P of X, which is an explicit density that we can optimize.
        5.3 Pros:
            Likelihood of training data gives good evaluation metric.
        5.4 Con:
            Sequential generation, so very slow.
        5.5 define a tractable ensity function, right using the conditional distribution, then optimize the likelihood of the training data.

## Variational Autoencoder

    Category: generative Model -> explicit density -> approximate density -> variational -> [VAE]

    1. How it works
        1.1 define an intractable density function. Model this with an additional latent variable Z.
        1.2 For 'AutoEncoder', 

## GAN

    Category: generative Model -> implicit density -> direct -> [GAN]

