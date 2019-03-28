# March 22nd meeting notes

## current strategy
    
    1. Currently target at some workshot around the end of the summer.
    
    2. We use marginal distribution for our first point and then use the joint distribution for the others.

    3. We use Bayesian Inference to do the factorization: casting the joint distribution as a product of conditional distributions.

    4. We learn from day 1, 10 users, by assuming all days in a week are alike. (Current step; Later may have more horizontal dependency in the table like distinguish weekdays and weekend)

    5. Using loosely-coupled degisn in our code so that users can use command line to input the parameters, ex. users number or how many days to generate. We output each user in a file and all days of that user are in that file.

## tasks next week

    1. Concerning the time windows like a effective range in a image, we can get P(xi | xk*...*xi-1). Assumption: Comparing to the image dependent on x1~xi-1, our case have dependent on time window. Futhermore, different field may have dependency like temporal: in a day, in 3-hours, or in half-day.

    2. Consider any additional dependency.

    3. Q: When generating multi-day overlapping window, how to transfer 1 day to another? A: 1 image is 1 window, then condider how to keep the dependency between windows.

    4. Consider long-term dependency, like same time 2 days back.

    5. Image -> homogeneous, all variables have same type.
       Our case -> different variable may have different type.
    Then maybe we can model different attibutes in different model.

    6. Image: joint distribution can be decomposed to product of elements in a matrix.
       Our case: joint distribution can be decomposed to product of elements in a matrix

    7. To check pixelCNN++ for channel operation and how to do multi-scale context.

    8. After modeling the #B, we can consider attributes like (UDP/TCP/ICMP) etc.

    9. To write down our formula like (1) in pixelRNN paper.