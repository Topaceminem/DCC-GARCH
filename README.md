# DCC-GARCH

DCC-GARCH is a Python package for a bivariate volatility model
called Dynamic Conditional Correlation GARCH,
which is widely implemented in the contexts of finance.

The basic statistical theory on DCC-GARCH can be found in
[Multivariate DCC-GARCH Model (Elisabeth Orskaug, 2009)](https://core.ac.uk/download/pdf/52106361.pdf).

Since my module DCC-GARCH is intially designed for
the computation of 
[SRISK (Brownlees & Engle, 2016)](https://academic.oup.com/rfs/article/30/1/48/2669965),
it only performs a Dynamic Conditional Correlation of order (1,1) and a GARCH of order (1,1).
However, empirical works find that DCC(1,1)-GARCH(1,1) is adequate in most of the financial problems
so the inconvenience may be minor.

In addition, a multi-step Monte Carlo simulation is also provided
for computing the expectation of one variable conditioning on the value of the other.

**Note that some parts of the code are still experimental,
as we haven't implemented public API for them.**
If you find a bug or have useful suggestions, please feel free to 
open an issue / pull request, or email [Suoer Xu](mailto:aterlioth@gmail.com).
Your contributions would be greatly appreciated!

## Dependencies

DCC-GARCH depends on numpy, scikit-learn and scipy.
Currently, it is only tested on Windows with Python 3.6.

## License

DCC-GARCH is distributed under the Apache License, Version 2.0.
