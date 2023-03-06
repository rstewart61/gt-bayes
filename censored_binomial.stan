data {
   int<lower=0> n;
   int<lower=0> M;
   int<lower=0,upper=n> ratings[M];
   int<lower=0,upper=1> right_censored[M];
 }
 parameters {
   real p;
 }
 model {
   // https://www.briancallander.com/posts/survival_models/censoring.html
   p ~ uniform(0.01, 0.99);
   for(i in 1 : M) {
        if (right_censored[i]) {
            target += binomial_lccdf(ratings[i] | n, p);
        } else {
            target += binomial_lcdf(ratings[i] | n, p);
        }
   }
 }

