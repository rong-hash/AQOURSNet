# Shapelet in Evolution

## Shapelet: [Ye, 2009](https://doi.org/10.1145/1557019.1557122) ([Website](http://alumni.cs.ucr.edu/~lexiangy/shapelet.html))
- Extract candidates by a **sliding window** through every time series in the dataset
- Distance from shapelet to series = *minimal* distance from shapelet to subsequences
- **Entropy** represents "discriminative power": $I(\textbf{D}) = -p_A \log p_A -p_B \log p_B$
- `check_candidate`: insert each series into `objects_histogram` by distance -> split, entropy, info gain
- Classify the series using **binary decision tree**: each shapelet provides a split

## Shapelet with pruning: [Ye, 2011](https://doi.org/10.1007/s10618-010-0179-5) ([Website](http://alumni.cs.ucr.edu/~lexiangy/shapelet.html))
- **Subsequence distance early abandon**: stop once the partial result exceeds the minimum
    - Deprecates with vectorization
- **Admissible entropy pruning**: optimistically place the series yet to be considered onto the distance axis

## Logical shapelet: [Mueen, 2011](https://doi.org/10.1145/2020408.2020587) ([Website](http://alumni.cs.ucr.edu/~mueen/LogicalShapelet))
- **Caching distance computation**: store $\sum x$, $\sum y$, $\sum x^2$, $\sum y^2$, $\sum xy$; $\mu_x = \frac{\sum_{u}^{u+l-1}x}{l}$, $\sigma_x = \frac{\sum_{u}^{u+l-1}x^2}{l} - \mu_x^2$
- **Candidate pruning**: shapelets similar to bad shapelets must also be bad
    - Triangular inequality: points on $v_2$'s orderline has a "mobility" within [-$R$, $R$] from $v_1$'s, $R$ = $\textrm{dist}(v_1, v_2)$

## Shapelet transform: [Lines, 2012](https://doi.org/10.1145/2339530.2339579) ([Website](https://sites.google.com/site/shapelettransform)) - aided by [Hills, 2014](https://doi.org/10.1007/s10618-013-0322-1)
- **Alternative shapelet quality measures**: 
    - *Kruskal-Wallis* & *Mood's median*: test whether two samples originate from distributions of the same median
    - *F-statistic*: higher value means greater inter-group variability compared to intra-group
- **Shapelet cached selection**: For every series, assess the candidate shapelets with *F-statistic of fixed-effect ANOVA*
    - Sort the candidates by quality, remove *self-similar* (overlapping) ones, and retain the top $k$
- **Length parameter approximation**: Randomly select 10*10 shapelets with ^, sort, take 25%~75% length
- **Distance-based data filter**: time series -- (distance to each shapelet) --> features with reduced dimensionality

## Unsupervised shapelet: [Zakaria, 2012](https://doi.org/10.1109/ICDM.2012.26) ([Website](https://sites.google.com/site/icdmclusteringts))
- Starting at the first series, calculate **split point** $\textrm{dt}$ and **gap score** $\textrm{gap} = (\mu_B - \mu_A) - (\sigma_A + \sigma_B)$
    - Iterating through every possible subsequence [$O(l^2)$] and $\textrm{dt}$ [$O(l)$] lowers the speed! 
- Store the *U-shapelet* with maximal $\textrm{gap}$ (along with the best $\textrm{dt}$)
- Points on the distance axis ("orderline") and left to $\textrm{dt}$ forms the set $D_A$; stop if $|D_A| = 1$
- **Clustered datapoint pruning**: remove series with distances smaller than $\mu(D_A) + \sigma(D_A)$ (more selective)
- Set the series with maximal $\textrm{dt}$ to be examined next, and repeat the algorithm

## Fast shapelet: [Rakthanmanon, 2013](https://doi.org/10.1137/1.9781611972832.74) ([Website](http://alumni.cs.ucr.edu/~rakthant/FastShapelet))
- **SAX discretization**: transform the time series into a symbolic representation
- **Random masking**: brute force search is still quadratic, and have *false dismissals* (amplifies tiny differences)
- **Hash signatures to objects**: objects similar in the original space have a high probability of collisions

## Learning shapelet: [Grabocka, 2014](https://doi.org/10.1145/2623330.2623613) ([Website](http://www.uea.ac.uk/computing/machine-learning/shapelets))
- Finally, ML enters the battlefield! 
- Linear model $\hat{\textbf{Y}} = \textbf{A}\textbf{X} + \textbf{B}$, regularized logistic $\textrm{loss} = -\textbf{Y} \ln \sigma(\hat{\textbf{Y}}) -(1-\textbf{Y}) \ln (1-\sigma(\hat{\textbf{Y}})) + ||\textbf{W}||^2$
    - SGD optimization
- **Differentiable soft-minimum function**: $m \approx \frac{\sum_j d_j e^{\alpha d_j}}{\sum_j e^{\alpha d_j}}$ (smaller $\alpha$, more precise the result)

## Shapelet forest: [Patri, 2014](https://doi.org/10.1109/BigData.2014.7004344)

## Shapelet tree: [Cetin, 2015](https://doi.org/10.1137/1.9781611974010.35)

## Random shapelet: [Wistuba, 2015](https://doi.org/10.48550/arXiv.1503.05018)

## Random shapelet forest: [Karlsson, 2016](https://doi.org/10.1007/s10618-016-0473-y)

## Numerical optimization: [Hou, 2016](https://doi.org/10.1609/aaai.v30i1.10178)

<br>

# Our Proposed Approach
### `Time2Graph` is, at its core, a **Finite State Machine**
- Transfer between states <==> Transformation between shapelets

### How to extract the shapelets from a subsequence rapidly?
- **Current method**: Use the distance between shapelets and subsequence to represent possibility. 
- **Our solution**: Use a neural network to *learn* the shapelet directly. 

### Notations

$v_i$ - the $i^{\textrm{th}}$ candidate shapelet

$t_j$ - the $j^{\textrm{th}}$ subsequence

$p_j(v_i)$ - possibility of a candidate shapelet mapping to a subsequence

### Algorithm Outline
1. **Unsupervised step**: Use *contrastive learning* to train a model to extract features from a subsequence
2. Using the traditional distance method, map the shapelets to the subsequences to get pairs $\{t_j, p_j(v_i)\}$
3. **Supervised step**: Concatenate an MLP after the unsupervised model. Use those pairs ^ for fine-tuning. 
4. The neural network is then ready. For each time series, we can go on to transfer its subsequences into possibility vectors and contruct the state transfer graph ("FSM Diagram").

### Another path
1. Graph Learning Method: **GIN** (Graph Isomorphism Network)
2. Make use of different time scales: 1 month, 3 months, 6 months
3. Use deep learning to learn the shapelets (as proposed above)
