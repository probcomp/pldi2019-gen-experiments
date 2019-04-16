# Body pose inference in Gen

To train the custom proposal distribution:
```
JULIA_PROJECT=. julia train.jl
```

To run importance sampling on a test image:
```
JULIA_PROJECT=. julia generate_gen_results.jl
```
