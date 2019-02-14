# Feb 8 Meeting Keynote

## Conclusion

After we have the baselines, we can try VAE or other advaced approaches.
For data, also many users in one characteristic group could be merge later.

## Job 1

conduct log transformation for the #B. (similar techniques are like square root, but log may be better)

## Job 2

use cross-validation to decide the number of components of the gaussian mixtures, like 1-4 fold (4 for fitting, 1 to detect likelyhood) This approach may be better than AIC/BIC.

## Baseline 1, gen col sepa.

\#B, delta time and DIP. DIP could be addressed later. Currently, we can discrete and index the data and then try to find unique address based on frequence.

## Aggregation

To build the histogram, how long should the bin-width be? Usually there are two solutions. The one is use cross-validation to decide. The other one which is proper for our current stage should be the heuristics, like bin-width wiki, such as square root of the #bin, #points.

## Factorization

B -> T
-> B-1

where, B_-1 means the last row generated.

## Survey on the two papers and [35] reference.

Do they assume rows are independent? We assume row have the dependency.

Read [35], how do they evaluate their model and how to design validation experiments.

## Stats

most sa [(10667863, '42.219.156.211'), (10664359, '42.219.156.231'), (5982859, '42.219.159.95'), (3995830, '42.219.153.191'), (2760867, '42.219.155.28'), (2126619, '42.219.153.62'), (2031099, '42.219.159.85'), (1740711, '42.219.158.156'), (1366940, '42.219.153.7'), (1342589, '42.219.153.89')]
process time 797

most da [(10445235, '42.219.156.211'), (10048082, '42.219.156.231'), (4011124, '42.219.153.191'), (2931129, '42.219.155.28'), (2560095, '42.219.159.85'), (2130010, '42.219.153.62'), (1740128, '42.219.158.156'), (1686586, '42.219.155.56'), (1420776, '42.219.153.89'), (1381596, '42.219.153.7')]
process time 422
