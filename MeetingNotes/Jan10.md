# Jan 10 Meeting Keynote

## what is the problem
User data or privacy data are hard to get as well as share. Before, researchers acquire such data by gathering real captured data or simulating a network.

Here we consider the data with the following characteristics — first, it's a categories data of anomaly and normally. Second, it's a relational data, such as joining & leaving of the user profile. Third, we hope to generate all these data independently.

We'll try to model the user attributes and the user graph (nodes and edges) instead of to model the whole traffic network because users are relatively simple to model and clear to control the distribution.

`given network flow => model user => generate field`

## why the problem is important

1. The high dimensional network traffic data are diverse and valuable, which could ask for a large number of fields.  

2. It's different from the traditional network simulation, which asks for the participation of the security experts and long term survey work. This work has three advantages. First, it could generate without expertise and only based on data from the enterprise. Second, it could do all process together and is easy to use. Third, do not use network simulator directly.

3. Some quality could be expected, such as less complexity, better generation. How to validate the result is remain a question to think.

4. Applications like defenders could be built based on the data we generate.

## potential approaches

1. Some VT data (traffic) could be available in the future.
2. GANs/encoder-decoder
3. Try to model low dimensional embedding of anomaly data
4. Deploy DNN models on network/email data
