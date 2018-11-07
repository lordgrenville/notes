# Maths

## Stats - basics

Correlation is the covariance normalised (<img src="https://rawgit.com/lordgrenville/notes/None/svgs/686181e29d7fbbadf06fab691a187e18.svg?invert_in_darkmode" align=middle width=172.923927pt height=46.8988773pt/>). Just covar between 2 variables in a way that's more resistant to changes in units.

MLE is an analytical way of saying what is the most likely parameter given this data? MAP is same, but incorporating the prior. For each dist, there's a corresponding way to represent the prior - this is called the conjugate prior distribution.

## Stats with Nadav

#### Stats for Data Science

Why do we need stats? To turn data into knowledge - if there's randomness, if not enough data, etc

Data can be structured - with labels - or not. If not you need to categorise it, but this is a lot of work so be judicious

Semi-structured. A mixture. ie an email

Data can be numeric (discrete/continuous), categorical, ordinal (it has an order) ie days of the week, rankings, arbitrary

Expectation is the mean. Variance is also a mean: it's the mean of the square distance from the mean

Covariance is also a mean (annoying, but what can ya do). It's the connection between 2 variables

(Pearson's) correlation is normalized covariance

Corr is always between -1 and 1, the former high inverse corr and the latter high corr

Lack of corr doesn't mean independence. It means lack of linear independence (like <img src="https://rawgit.com/lordgrenville/notes/None/svgs/66c4f18169c7e70ca51aee17e5f937ad.svg?invert_in_darkmode" align=middle width=48.18103454999999pt height=21.18721440000001pt/>, but nonlinear would be <img src="https://rawgit.com/lordgrenville/notes/None/svgs/bcb037123f40331d39ba394f2f25c697.svg?invert_in_darkmode" align=middle width=46.514371199999985pt height=26.76175259999998pt/> or something)

Presence of correlation *does* imply dependence  

#### Models

A model is a way of looking at the data <img src="https://rawgit.com/lordgrenville/notes/None/svgs/bc737115ca0cc836cabcebcc1ad950ee.svg?invert_in_darkmode" align=middle width=80.61497069999999pt height=26.76175259999998pt/> (tilde means distributed as, the Normal distribution is the Gaussian bell-curve)


Shoes are not normally distributed since normal goes from <img src="https://rawgit.com/lordgrenville/notes/None/svgs/f7a0f24dc1f54ce82fecccbbf48fca93.svg?invert_in_darkmode" align=middle width=16.43840384999999pt height=14.15524440000002pt/> to <img src="https://rawgit.com/lordgrenville/notes/None/svgs/1d5ba78bbbafd3226f371146bc348363.svg?invert_in_darkmode" align=middle width=29.223836399999986pt height=19.1781018pt/>, shoes don't! 

So models aren't reality, but they're helpful ways to think about sutff 

A **parametric model** is a model with paramaters - so in the shoes it might be mean and variance, or in a Bernoulli model ...

Start simple, increase complexity as you go

Likelihood function calculates how likely it is that the data is correct, given the model

We always assume that the data is IID. We can almost never prove it but assume it.

<img src="https://rawgit.com/lordgrenville/notes/None/svgs/0c19c31ee0328a57bc8447f564bd260e.svg?invert_in_darkmode" align=middle width=98.34678809999998pt height=24.657735299999988pt/> 

#### Maximum Likelihood Estimator

arg max <img src="https://rawgit.com/lordgrenville/notes/None/svgs/1e072bacc57819de27e20364be98bffd.svg?invert_in_darkmode" align=middle width=32.146202549999984pt height=24.65753399999998pt/>

NB MLE is the value of a test statistic (like mu or sigma) given a series of data points. We ask: GIVEN THIS DATA, WHAT IS THE LINE THAT FITS IT (regression)? and we answer, THIS IS THE MAXIMUM LIKELY MEAN (or whatever) FOR THE DISTRIBUTION. 

The mechanics work through derivatives of the log (just to make it easier to work with, but this is less important)

This is easy when finding the max/min is easy
Bernoulli is a binary distribution
We calculate the P~MLE~ for the Bernoulli and found it confirmed our intuition

#### Estimators
the standard error is the standard deviation of the estimator (the hat <img src="https://rawgit.com/lordgrenville/notes/None/svgs/d92f3abf67d0476f612d337a28bab188.svg?invert_in_darkmode" align=middle width=9.56628914999999pt height=31.50689519999998pt/> means estimated)

The estimator (MLE) tends toward <img src="https://rawgit.com/lordgrenville/notes/None/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> as <img src="https://rawgit.com/lordgrenville/notes/None/svgs/8e5ac09b149a8f375637b349458e91e1.svg?invert_in_darkmode" align=middle width=51.87587954999999pt height=14.15524440000002pt/> and is distributed normally

We can calculate a confidence interval for any confidence level.

## Lecture 2 - Bayesian Inference

<img src="https://rawgit.com/lordgrenville/notes/None/svgs/6c7f53922c1a1e673f6c1580827d3ad4.svg?invert_in_darkmode" align=middle width=58.55026814999999pt height=26.97711060000001pt/>

You have priors and posteriors, each of which have distributions. There are recipes: for each distribution, you have a prior distribution (Bernoulli <img src="https://rawgit.com/lordgrenville/notes/None/svgs/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode" align=middle width=16.43840384999999pt height=14.15524440000002pt/> Beta distribution, normal <img src="https://rawgit.com/lordgrenville/notes/None/svgs/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode" align=middle width=16.43840384999999pt height=14.15524440000002pt/> normal, etc.) If posterior and prior are from same family it is known as aa conjugate prior and we say that they are conjugate pairs (I think?) 

In linear regression, the superscript is the dimension number, and the subscript is the row number. 
We transform our data to a matrix. We multiply x by <img src="https://rawgit.com/lordgrenville/notes/None/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16555099999999pt height=22.831056599999986pt/>, so we have X and Y as matrices and <img src="https://rawgit.com/lordgrenville/notes/None/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16555099999999pt height=22.831056599999986pt/>s as a vector.

We can calculate the MLE for Y given <img src="https://rawgit.com/lordgrenville/notes/None/svgs/8b953bcf8311e0fa0ae2facaf3cdab8e.svg?invert_in_darkmode" align=middle width=31.466856299999993pt height=22.831056599999986pt/>. 



Basically: P of A or B is sum, of A and B is product

Flip 2 coins, prob of A being heads or B being heads is 0.5 + 0.5 =1? Can't be

Bayesian Venn diagram shows us the problem, we're double-counting since there's overlap (P of both happening) - so we need the Bayesian formula 

Likelihood (based on [this](https://en.wikipedia.org/wiki/Likelihood_function#Example_1) )

The Likelihood func is the likely prob based on observations

We often use the log-likelihood as it's easier to work with



- <img src="https://rawgit.com/lordgrenville/notes/None/svgs/64f9f73dde13102bac735f4b519f0b7e.svg?invert_in_darkmode" align=middle width=138.25115039999997pt height=46.8988773pt/> Or in English, the probability of  seeing A given B is the probability of seeing them both divided by the  probability of B.

- <img src="https://rawgit.com/lordgrenville/notes/None/svgs/199587dd5e3a1c90ca8b2a5f3f506048.svg?invert_in_darkmode" align=middle width=138.25115039999997pt height=46.8988773pt/> Or in English, the probability of  seeing B given A is the probability of seeing them both divided by the  probability of A.

  Thus:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/dd3bbf0d7f7ea5fecbc291905c85d738.svg?invert_in_darkmode" align=middle width=241.45544115pt height=24.65753399999998pt/>

  Which implies:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/81d966766421a78c2279ca83736e5412.svg?invert_in_darkmode" align=middle width=188.98746239999997pt height=46.8988773pt/>

  And plug in <img src="https://rawgit.com/lordgrenville/notes/None/svgs/1ed31ca923d9d78dc54cbc98cdd45c15.svg?invert_in_darkmode" align=middle width=8.21920935pt height=14.15524440000002pt/> for <img src="https://rawgit.com/lordgrenville/notes/None/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/lordgrenville/notes/None/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> for <img src="https://rawgit.com/lordgrenville/notes/None/svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/>:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/3a22fd1cc7bc4c53fa576a4debd9cadf.svg?invert_in_darkmode" align=middle width=179.75217479999998pt height=46.8988773pt/>

  Nice! Now we can plug in some terminology we know:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/cda99c245151d1984e42a9b381a1d98e.svg?invert_in_darkmode" align=middle width=221.01649679999997pt height=45.072403200000004pt/>

  But what is the <img src="https://rawgit.com/lordgrenville/notes/None/svgs/908a36b87027835ac068979bac2eced3.svg?invert_in_darkmode" align=middle width=48.29346224999999pt height=24.65753399999998pt/> Or in English, the probability of our data?  That sounds weird… Let’s go back to some math and use B and A again:

  We know that <img src="https://rawgit.com/lordgrenville/notes/None/svgs/90ef7ba1e751ba020c359f4239bab1e3.svg?invert_in_darkmode" align=middle width=131.7123291pt height=24.65753399999998pt/> (check out this [page ](http://en.wikipedia.org/wiki/Marginal_distribution)for a refresher)

  And from our definitions above, we know that:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/0c47ab852d6b4293aa8e2598341de4ca.svg?invert_in_darkmode" align=middle width=189.75452429999999pt height=24.65753399999998pt/>

  Thus:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/fdd398d9cec1b97d3777cf3156ed9fc6.svg?invert_in_darkmode" align=middle width=200.91895889999998pt height=24.657735299999988pt/>

  Plug in our <img src="https://rawgit.com/lordgrenville/notes/None/svgs/1ed31ca923d9d78dc54cbc98cdd45c15.svg?invert_in_darkmode" align=middle width=8.21920935pt height=14.15524440000002pt/> and <img src="https://rawgit.com/lordgrenville/notes/None/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/>:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/b0489f8a639b5198f92aca09f11ebeb8.svg?invert_in_darkmode" align=middle width=192.5683023pt height=24.657735299999988pt/>

  Plug in our terminology:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/529f9a3897d80cb4bf3ea517f7f5c08e.svg?invert_in_darkmode" align=middle width=215.62552440000002pt height=24.657735299999988pt/>



  Wow! Isn’t that awesome! But what do we mean by <img src="https://rawgit.com/lordgrenville/notes/None/svgs/b0dc34367b5a2d0cf8d8a49287b0accf.svg?invert_in_darkmode" align=middle width=8.21920935pt height=14.15524440000002pt/>. This means to  sum over all the values of our parameters. In our coin flip example, we  defined 100 values for our parameter p, so we would have to calculated  the likelihood * prior for each of these values and sum all those  answers. That is our denominator for Bayes Theorem. Thus our final answer  for Bayes is:

  <img src="https://rawgit.com/lordgrenville/notes/None/svgs/00b5242f2449a47aa058f3fb66ea59d4.svg?invert_in_darkmode" align=middle width=248.54497469999995pt height=45.072403200000004pt/>
  
#### Hypothesis testing

for each point calculate prob came from norm then get proportion of false/rejected null hyp that's the pop p value

P-value is the prob of everything that is at least as unlikely given H0, and is more than likely given H1

p value is about protecting yourself from a very specific error, that is an error you can point to - not any kind of error. That's why we used one-tail test - we're only looking to compare H0 and H1.

R is the rejection region, the areas for which the p-value is below the threshold - the values for which we reject the hyopthesis. In other words, instead of specifically calculating a sample for p-value, we're comparing the entire sample (I think this is the first HW question?)

If you add two IID normally distributed variables, the sum ahs the sum of the means, the product has the product of the means.

Phi is the CDF of the normal dist. Instead of calculating it, there are tables. C = Zalpha / n is basically the formula we need

If we know the variance - Z-test; if not t-test

## Back to basics: explaining stuff in my own terms

MLE WTF: 

<img src="https://rawgit.com/lordgrenville/notes/None/svgs/076780018d207075dc328f80b5f2e89c.svg?invert_in_darkmode" align=middle width=204.00929669999996pt height=24.65753399999998pt/>

Likelihood is to parameters given data what probability is to data given parameters (the semicolon means "given")

The difference between Bayes and frequentist is that f says nothing wihout data, once given data all we can say is "this data comes from distribution X". B says we first take a prior (could be wild guess, but often *we do have a sense* of what it should be), then take data and add that on, then you get a posteriori. So Bayesian equivalent of MLE is MAP.

MLE is at the core of statistical modelling b/c this is what we do day to day in DS: we're given data, and we map it to a model. 

Technically, though MLE asks "how likely is the data given the model with parameter <img src="https://rawgit.com/lordgrenville/notes/None/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>"? Why not just ask how likely is the parameter? Answer is that parameter is not an RV: the data is. 

Confidence level + alpha = 1

(Alpha is complement of CL)

If p-value less than alpha, reject null hypothesis

A high alpha level will reject a lot of true hypotheses (Type I error - reject true H<sub>0</sub>), whereas a too small alpha will accept a lot of false hypotheses (Type II error - not rejecting a false hypothesis)

## Statistical tests

We can do z-tests, t-tests, Chi<sup>2</sup> tests (likelihood ratio). The LR is the same as (the term also has a few other meanings in different contexts). Say we want to compare pop mean vs mean of a sample. We could do a z-test, but could also do a LR test. We "approximate the distribution of the "normalized" likelihood ratio (that is, likelihood divided by the maximum likelihood) by a Chi<sup>2</sup> distribution and proceed to find what values of the population mean are consistent with that statistics ([source](https://math.stackexchange.com/questions/882393/maximum-likelihood-estimate-vs-likelihood-ratio-tests)).

# Useful SQL commands

SHOW COLUMNS from TABLE;  <- get column names

ALTER TABLE <- to rename column, you can use FIRST to add a first column (or AFTER)

DELETE from TABLE where COLUMN_NAME is NULL LIMIT 10; <- (deletes rows, not cols, also note "is", not =)

needs a limit otherwise won't run in safe update mode

ALTER TABLE DROP columnname < - 

# Numpy/Pandas

rows before columns, always! <img src="https://rawgit.com/lordgrenville/notes/None/svgs/3a50eb5b4bb346ec8c716e8298cbdb16.svg?invert_in_darkmode" align=middle width=30.607056149999988pt height=22.465723500000017pt/> means row i, column j

NEVER USE PANDAS METHODS AS FIELD NAMES

Getting a single column can only be done by column name. Slicing by number requires begin and end point

Slicing numpy arrays - use commas to separate dimensions - so slicing a tensor be like (:3, 2:4, 1:5) or whatevs

Pandas - if you want to slice a middle dimension, but not an earlier one, use : so df.loc[:, 2:4, 1:5]

Boolean indexing is on the whole array/DF

Matplotlib line styles - https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html -- .- etc

sort_index sorts by the index (if primary key, not so interesting. sort_values sorts by values in a column

df.where is boolean indexing - replace where true

Shift tab for help on method, do this a lot!

Axis = row/column (0,1)

matplotib: naming the plot means you're going to address it
so plot(1)
some stuff
plot(2)
soe stuff
plot(1)
more about 1
etc

slicing pandas DFs - iloc is integer-location (going by index number), or loc for name-based locations 

value_counts() is a good way to count

to change values - use loc/iloc to move, and 'at' to *change*

arange gives you ranges. you can find max, min, mean by columns (axis=1 for by row)

For a 2D array, numbers.min() finds the single minimum value in the array, numbers.min(axis=0) returns the minimum value for each column and numbers.min(axis=1) returns the minimum value for each row.

Python zip converts row vectors to column vectors...
A = [1,2]
B = [3,4]
C = [5,6]
for i in zip(A,B,C):
	print i
(1, 3, 5)
(2, 4, 6)
*it's a transpose!*

np.any, np.where are your friends - quick true-false search/boolean masking

Pandas note: if you want just the pandas valus that fit a certain condtion, it's df[df.fart==True]. Using df.where(df.fart==True) will return a df of the same size, with NaNs in place of your values

np.newaxis adds an axis, turning a 2D array into a 3D for eg. But also valuable to turn an arange(100) with shape (100,) into a vector with shape (100,1). Remember, an array of numbers in not automatically a 2D vector (of n by 1), so if you're getting NuPy unable to broadcast issues, make sure they're both these.  

np.random.normal -> draw from normal with loc,std = <img src="https://rawgit.com/lordgrenville/notes/None/svgs/ad5bfc709b2e6c94b4dab37e68111117.svg?invert_in_darkmode" align=middle width=27.19370774999999pt height=14.15524440000002pt/> ; np.random.uniform -> draw from intervals at random uniformly

df.plot is awesome, just remember if you have more than one column and want to plot X vs Y, give column names as a string, not columns themselves! So not df.plot(df.foo, df.bar); but df.plot('foo', 'bar')!! 

Annoying pandas stuff: df['a'] selects col A; df['a', 'b'] returns an error...you need df[['a', 'b']]

## Numpy/LinAlg/MatLab Stuff
There are two ways of multiplying two vectors: elements-wise, or dot product. One multiplies, the other multiplies and sums. The former is `np.multiply`, the latter is `@`. An example. Say we use a perceptron to predict a linear classifier (a straight line diving the A circles from the B circles). It may give us weights (coefficients to multiply our xes by) and a scalar term called an intercept (so <img src="https://rawgit.com/lordgrenville/notes/None/svgs/a7e2245854c13c7b31d5beae05d04cae.svg?invert_in_darkmode" align=middle width=191.95967174999998pt height=21.18721440000001pt/>). If you have an array of xes and want to get ys, you do `coefs@xes + intercept`. Numpy magic! 

# Matplotlib

plt.figure() opens a figure
everthing will be held there until you say show(), or open a new figure!

## General DS stuff you were too embarassed to ask

Classification is when data is categorical
Regression is when data is numeric

Overfitting or high variance is when the model captures noise as well as signal - it fits the training data too well.

We can respond to this in one of two ways - feature reduction (eg PCA), and regularisation. The latter is when you have a lot of *slightly* useful features.

Cost function - this is the function judging how wrong you are. You use <img src="https://rawgit.com/lordgrenville/notes/None/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> 's to weigh the different factors.

ROC Curve - super simple method, only for binary classifiers. You graph your TPR and FPR: improvements in the one will generally come at the cost of the other. The 50/50 curve runs down the diagonal - if this is what you get, may as well flip a coin. The y-axis is "always yes", x-axis is "always no". If you're in the upper triangle of the plot then your model is doing well.

Bias = underfitting; variance = overfitting

### Notes from Andrew Ng
Thetas in a cost function: <img src="https://rawgit.com/lordgrenville/notes/None/svgs/aee5a3976601efd4227faff1a7fbb9f3.svg?invert_in_darkmode" align=middle width=247.97737304999995pt height=24.65753399999998pt/>
For multi-variable regression (predicting house prices with multiple features) it's the same process as with one variable. 
If our values are wildly disparate, this can cause havoc with the gradient descent, so we do **regularisation**: 
"Two techniques are feature scaling and mean normalization. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just 0. For both, adjust your input values as shown in this formula:<br>
<img src="https://rawgit.com/lordgrenville/notes/None/svgs/7080e33c9746adcb3eeec68444e28b62.svg?invert_in_darkmode" align=middle width=93.6608838pt height=41.4194451pt/>

Where <img src="https://rawgit.com/lordgrenville/notes/None/svgs/ce9c41bf6906ffd46ac330f09cacc47f.svg?invert_in_darkmode" align=middle width=14.555823149999991pt height=14.15524440000002pt/> is the average of all the values for feature (i) and <img src="https://rawgit.com/lordgrenville/notes/None/svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.35637809999999pt height=14.15524440000002pt/> is the range of values (max - min), or <img src="https://rawgit.com/lordgrenville/notes/None/svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.35637809999999pt height=14.15524440000002pt/> is the SD.

We don't need to do gradient descent - we can do normal equation and solve the entire matrix at once. This is more accurate and doesn't have as much room for error, but more computationally expensive. In practice at about 10k might want to switch methods.

#### Logistic Regression (week III)
This is where we get into classification. We want to map everything to between 1 and 0 (yes or no), so in practice we use a sigmoid function <img src="https://rawgit.com/lordgrenville/notes/None/svgs/0caf806368fb2ab367d0108eafb2a22a.svg?invert_in_darkmode" align=middle width=53.81279145pt height=43.42856099999997pt/> (where z is <img src="https://rawgit.com/lordgrenville/notes/None/svgs/3c4326653e9d61030bfd06a36a56acdf.svg?invert_in_darkmode" align=middle width=27.92410829999999pt height=27.6567522pt/>). It gives us the probability (if it's 0.7, there's 0.7 probability that yes)

Dealing with overfitting either by 1) reducing the number of features: (manually select which features to keep, use a model selection algorithm), or 2) Regularization (Keep all the features, but reduce the magnitude of parameters <img src="https://rawgit.com/lordgrenville/notes/None/svgs/455b7e5df6537b98819492ec6537494c.svg?invert_in_darkmode" align=middle width=13.82140154999999pt height=22.831056599999986pt/>. Works well when we have a lot of slightly useful features.)

Regularisation: sometimes you want it to be quadrati, but you don't want the <img src="https://rawgit.com/lordgrenville/notes/None/svgs/ca51fbbd81047b86524d9877e4ce2ce4.svg?invert_in_darkmode" align=middle width=40.02286529999999pt height=26.76175259999998pt/> to have too much weight. To counter this, what if we added a condition that our cost function find the lowest cost for (cost function) + 1000x<sup>4</sup>? That way we'd need to make sure our higher x had a very low impact.

We formalise this with the term <img src="https://rawgit.com/lordgrenville/notes/None/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>.

### Neural Nets (weeks 4-5)

OK, so same as before but we're scaling up. We now have a network made of "layers" of "nodes" - where each of the xes goes into each node, and is multiplied by the weights (thetas), plus the bias unit. The value for each node in each layer is the "activation function" of (all the inputs * all the weights) + bias. Each layer has a matrix of weights: enough weights for all the xes, for all the nodes in the layer.  If layer one has 3 noes and layer 2 has 4, the weights for layer 1 ill be **4 x 4**, because you add one for the bias in the **current layer**, but not for the output to the next layer.

To reiterate: with fully-connected layers, your weights <img src="https://rawgit.com/lordgrenville/notes/None/svgs/6ae66a3cc3c796c20bca79813e802df0.svg?invert_in_darkmode" align=middle width=20.958959999999987pt height=24.65753399999998pt/> matrix has rows = xes + 1, and cols = # of nodes

Multi-class classification has multiple nodes in the output layer - one for each x

Backpropagagtion involves finding the partial derivatives (the delta, <img src="https://rawgit.com/lordgrenville/notes/None/svgs/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode" align=middle width=7.928075099999989pt height=22.831056599999986pt/>) of theta for each node in each layer, and then summing these in a a matrix (big delta, <img src="https://rawgit.com/lordgrenville/notes/None/svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/>). We don't find error for the first layer b/c it's our input data.

Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)

# Data Modeling 

####  Some real basics
Supervised learning is classification/regression: we have labelled data and some unlabelled that we want to predict.
Unsupervised learning is modelling/clustering: you have no labels, tring to find a shape.
Reinforcement: we haven't gotten to it - watch this space...

### Lecture I

Bias is how far away you are, variance is how big the error range is. See slides for examples. Neither is good, and they are to some extent reversely proportionate. Best is a middle point.

Linear Regression

for continuous data. For the simplest case (univariate), we use the mean squared error - this way we penalizes much more for farther deviations

Gradient descent is based on the idea that as you move away from a max/min point, your rate of change increases. So we want to move away from this, we keep moving and comparing the difference in steps, minimising them, until we get to the smallest local move - that's a local minimum

Univariate: Multivariate :: House prices by area:House prices by area, age, location...

regularisation is modifying the impact of larger factors in polynomial expressions (we want our higher power thetas to have more of an impact) we multiply it by a lambda

Linear regression is for a continous outcome what is y's connection to x. Logistic regression is for categorical applications: yes or no; a, b or c

One form of regularisation is ridge regression/LASSO. Here's the deal with these (mostly taken from [here](https://en.wikipedia.org/wiki/Tikhonov_regularization)). You want to do regression using your common or garden Ordinary Least Squares cost function. But the solution is not unique (we call it an ill-formed problem), giving you under- or over-fitting. In this case, we add an extra term which we try to minimise. The goal is to remove some of the factors, get rid of the ones that contribute less and find the ones that count most. L1 uses absolute value, L2 uses the square (Euclidean distance or some such). Either way it is related to the "norm" - the "size" of the vector.

![image.png](attachment:image.png)

### Lecture II -  Binary Classfiers

Not all errors are equal, for eg in  medicine false positive not equal to false negative
the famous "confusion matrix" quadrant
in autonomus driving, true negatives might be infinite. they will skew everything and are not very interesting
so you get all kinds of alternative methods of weighting them: not just +1 for TP, TN and -1 for FP, FN, but for eg:
False Discovery Rate: how many of ur yeses were wrong?
Miss Rate: how many of the actual yeses did you miss?
more here: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context) 
so we have tradeoffs
we can maximise in bizarre ways (always return True for eg), but everything depends on situation

AUC of the ROC curve is a common measure in the ML community, the idea is that the integral (area under the ROC curve) is the likelihood that we will rank a random positive higher than a random negative

F-score is the harmonic mean (a special kind of mean, pretty siple, goes back to Pythagoras) of precision and recall

Part 2 of the lecture is about a Naive Bayesian classifier. We calculate the MAP, which is basically what your best guess is taking into account your prior and posterior probabilities. Mathematically it is about maximising the argument - it's the mode (highest point) of the posterior distribution

A note about Naive Bayes: how do we go from Bayes *theorem* to **classifier**? Answer is that given p of Y (later event) and of X (earlier), if we want to calculate P(Y|X) we use bayes theorem. But if we want P(X|Y) - in other words, given the fact, what is the most likely probability that produced it? - this is called finding the posterior. That's what a Bayesian classifier does. It's naive if it assumes that every dimension (column, all 784 pixels) is independent, so you can just multiply the probabilities to get the total probability. 

In gradient descent, we need to choose <img src="https://rawgit.com/lordgrenville/notes/None/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> (step size) in such a way that we get to the right local minimum. If it's too big, we'll overshoot; too small, we'll take too long.

Naive Bayesian assumes that all features are independent (no correlation between them) - that's why it's naive. 

Quote from WP: "For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features."

For Gaussian naive Bayes, we must estimate the mean + variance for each feature of each class. then just plug and run...

logistic regression

probability of y or n based on continuous features 

we use the logit function, the invrse of the familiar S-shaped "logistic" curve (google it...)

great resource for PCA, if simplified: http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf (relevant bits are only a couple of pages)
this sumarises PCA as getting orthogonal eigenvectors of the covar matrix (in class we used a slightly more sophisticated method), sorting from biggest to smallest, choosing the top p vectors (optional) and recombining them into a matrix, and then multiplying by the data. The data will have p dimensions if we dropped some, but even if we don't PCA is transformative.

Unsupervised 2: Clustering

clustering is an alternative to PCA. we assign everything a cluster (arbitrarily at first), then iteratively measure distance from centroids of clusters to members and reassign them to nearest centroid.

# Clustering - Danny Barash

Ordinal means can be ordered - say, 1 to 5 stars
then there's numeric (simple)
and categorical (can't be ordered)

Supervised learning is for predictions (labeled data)
Unsupervised is for finding structure
Most data in the wild is unlabeled

We discussed kNN and k-means, hierarchical clustering

Distance function must be positive, symmetric (same from A to B and from B to A), and must fit the triangle inequality (dist between A and B can't be greater than distance between either and C)

There's also correlation distance, absolute, maximal

If we have big gaps between types of objects, (socks and computers) we should normalize

 

### Lecture 4 - SVD
A singular matrix is one that has a determinant of 0. **This means it cannot be inverted.**(The inverse is the matrix which gives *I* when multiplied by original matrix.)<br>
We can do something called the Moore-Penrose pseudo-inverse, using SVD.<br>
SVD can handle any matrix, even non-square, non-symmetric.<br>
We can break down (factorize) any matrix into <img src="https://rawgit.com/lordgrenville/notes/None/svgs/51f741b570394ab9406d6656b4388ce9.svg?invert_in_darkmode" align=middle width=84.49016564999998pt height=27.6567522pt/>, where U,V are orthogonal and <img src="https://rawgit.com/lordgrenville/notes/None/svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/> is diagonal.
U and V are called the singular vectors.<br>
U and V are orthogonal: that means that their transpose is also their inverse.<br>
They are also singular, this is almost the same, but for complex matrices, see [here](https://www.quora.com/What-is-the-difference-between-a-unitary-and-orthogonal-matrix)<br>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Singular_value_decomposition.gif/280px-Singular_value_decomposition.gif"></img>

Visualisation of a singular value decomposition (SVD) of a 2-dimensional, real shearing matrix M. First, we see the unit disc in blue together with the two canonical unit vectors. We then see the action of M, which distorts the disc to an ellipse. The SVD decomposes M into three simple transformations: a rotation V*, a scaling Σ along the rotated coordinate axes and a second rotation U. Σ is a diagonal matrix containing in its diagonal the singular values of M, which represent the lengths σ1 and σ2 of the semi-axes of the ellipse (from [Wikipedia](https://en.wikipedia.org/wiki/Singular_value)).<br>
<br>
U and V are inverse of one another.<br>
So if you can't get the inverse you can get the pseudo-inverse via SVD: <br>
> <img src="https://rawgit.com/lordgrenville/notes/None/svgs/0a9b5714f791dfb09438c984d0174a92.svg?invert_in_darkmode" align=middle width=86.68421684999998pt height=27.6567522pt/><br>
> <img src="https://rawgit.com/lordgrenville/notes/None/svgs/8bf7b98297959b311e664c9ecb07e106.svg?invert_in_darkmode" align=middle width=95.90796764999999pt height=27.6567522pt/>
<cr>For each value in the diagonal (D<sub>ii</sub>), you either take 1/prev, or if prev = 0, take 0.
    
You can run SVD on your input. If the matrix is almost singular (ie variables are very close to linearly dependent on others), then take out the near-zeroes otherwise it can hurt your data analysis.  

## Convolutional Neural Networks

"**The bread and butter of neural networks is _affine transformations_: a vector is received as input and is multiplied with a matrix to produce an output (to which a bias vector is usually added before passing the result through a non-linearity)**.  This is applicable to any type of input, be it an image, a sound clip or an unordered collection of features: whatever their dimensionality, their representation can always be flattened into a vector before the transformation. Images, sound clips and many other similar kinds of data have an intrinsic structure. More formally, they share these important properties:

-  They are stored as multi-dimensional arrays.
-  They feature one or more axes for which ordering matters (e.g., width and height axes for an image, time axis for a sound clip).
-  One axis, called the channel axis, is used to access different views of the data (e.g., the red, green and blue channels of a color image, or the left and right channels of a stereo audio track).

These properties are not exploited when an affine transformation is applied; in fact, all the axes are treated in the same way and the topological information is not taken into account. Still, taking advantage of the implicit structure of
the data may prove very handy in solving some tasks, like computer vision and speech recognition, and in these cases it would be best to preserve it. This is where discrete convolutions come into play.

A discrete convolution is a linear transformation that preserves this notion of ordering. It is sparse (only a few input units contribute to a given output unit) and reuses parameters (the same weights are applied to multiple locations
    in the input)." [source](https://arxiv.org/pdf/1603.07285.pdf)

_Pooling_ is basically like convolution - you pass a kernel over it, but instead of multiplying by a kernel weight, you do something else (like the average or the max), the point of it is to reduce the size of your input.

## Spilling all the t* on LSTMs
*<img src="https://rawgit.com/lordgrenville/notes/None/svgs/0b835be8ebf807a82ca68141300b476b.svg?invert_in_darkmode" align=middle width=81.02754329999999pt height=24.65753399999998pt/>

Some ideas from [DJ Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/):<br><br>In an RNN, each single neuron (in a regular network) is in fact a rolled up neuron that takes in all of the input (per batch). It then feeds each output (<img src="https://rawgit.com/lordgrenville/notes/None/svgs/30d8c0e4d071178d72d5520a22db405a.svg?invert_in_darkmode" align=middle width=73.69883564999999pt height=22.831056599999986pt/>) into the next one.  
<img src="images/roll.jpg"/>


More specifically, it cycles back into itself, and the unrolling is only a metaphor. Something like this: <img src="images/rnn.jpg" />

So instead of normal backpropagation we do backpropagation through time - ie with an added time element. But it's restricted becuase the computation quickly gets out of control, so in an RNN, each t only takes into accout a few ts backwards. Also, RNNs have the vanishing/exploding gradient problem - each time you take a sigmoid, you increase the likelihood that it will flatline! <img src="images/sigmoid_vanishing_gradient.png" />

#### LSTMs to the rescue
LSTMs do all this, but with a hidden state <img src="https://rawgit.com/lordgrenville/notes/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/>, that has four different "gates" (basically each one is just a linear combination with a different role - forget, remember, etc.). Without going into the details right now, this state is preserved, meaning that we can retain all of the info from a batch of input data. Thus LSTMs have become the most popular implementation of RNNs. 

#### Architecture
In Keras (we'll use the Sequential rather than the Functional API b/c that's how my boy Jason Bradley rolls, but they're very similar), you create an LSTM with the method: `model.add(LSTM(128), input_shape=(1, X_train.shape[1]))`<br> That is, 128 is the number of neurons in each layer, and `input_shape` is the size of the "unrolled" neuron (the amout of data to pass through your neuron). The `return_sequences` flag gives you the option to return all of the stages of the LSTM.

#### Early Stopping
Monitors a paramater (loss, val_accuracy) and ends early if it hasn't improved for n epochs 

### Pandas stuff I'm TIRED OF GOOGLING
Display rows with NaN: `df[df.isnull().any(axis=1)]`<br>
Search for text in text column: `df[df.text_column.str.contains('whatever')]`<br>
Adjust figsize for entire notebook: `plt.rcParams['figure.figsize'] = [12.0, 6.0]`<br>
Copy-paste pandas table from SO: `df=pd.read_clipboard(sep='\s\s+', engine='python'); df.head()`<br>
Group-apply-combine: `df.groupby(['col1', 'col2'])['col3'].mean()`<br>
Two conditions mask: col A is x, col B not NaN: `df[(df['A'] == '2013') & (~pd.isnull(df['B']))]`<br>
Split one column into two: `df['A'], df['B'] = df['AB'].str.split(' ', 1).str`
Subplots in Matplotlib: look [here](https://stackoverflow.com/a/38438533/6220759)!!!!!!!! (`subplot2grid`)
***

## [Guy on Reddit](https://old.reddit.com/r/datascience/comments/7cx9yt/how_to_learn_pandas/) with Strong Opinions
Once you have finished your first kernel, you can go back to the documentation and complete another section. Here is my recommended path through the documentation [(source)](https://medium.com/dunder-data/how-to-learn-pandas-108905ab4955):

    Working with missing data
    Group By: split-apply-combine
    Reshaping and Pivot Tables
    Merge, join, and concatenate
    IO Tools (Text, CSV, HDF5, …)
    Working with Text Data
    Visualization
    Time Series / Date functionality
    Time Deltas
    Categorical Data
    Computational tools
    MultiIndex / Advanced Indexing

### Relevant HN discussion [(link)](https://news.ycombinator.com/item?id=17075126)
For a large fraction of probability theory, you only need two main facts from linear algebra.

First, linear transforms map spheres to ellipsoids. The axes of the ellipsoid are the eigenvectors.

Second, linear transforms map (hyper) cubes to parallelpipeds. If you start with a unit cube, the volume of the parallelpiped is the determinant of the transform.

That more or less covers covariances, PCA, and change of variables. Whenever I try to understand or re-derive a fact in probability, I almost always end up back at one or the other fact.

They're also useful in multivariate calculus, which is really just stitched-together linear algebra.

reply:I think the first point is only true for symmetric matrices (which includes those that show up in multivariable calc). In general, the eigenvectors need not be orthogonal.

reply:Yep, you could well be right. The image of an ellipse under a linear transform is definitely an ellipse, but I'm not sure about the eigenvectors in the general case.

The symmetric case is by far the most relevant for probability theory though.

reply:In general it's the eigenvectors of the positive-semidefinite (hence symmetric) part of the left polar decomposition.
