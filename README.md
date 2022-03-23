# Probs

Draw samples from certain domain with various sampling techniques. 



## Usage

Construct a sampler of certain type 

```rust
use sampler::Sampler;
fn test() {
  sampler::univar::Icdf::new()
```

Sample a distribution with it 

```rust
    .sample(distribution::univar::gaussian::<i8>(0.0, 32.0))
```

Use it like any `Iterator` as you would 

```rust
    .enumerate()
    .for_each(|(i, x)| println!("sample#{}: {}", i, x))
}
```



## Distribution

**ANY** customized/wired distribution can be sampled with provided samplers

```rust
sampler::univar::Icdf::new()
  .sample(|x: &u8| (x % 8) as f64)
```

> Some sampler needs a burn-in period to achieve equilibrium



## Multi-dimensional

[Metropolis Sampler](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) is used to sample distribution from high dimension

- Fix-dimensioned domain is represented with [ndarray](https://crates.io/crates/ndarray)

```rust
sampler::multivar::Metropolis::<f64, nd::Ix1>::new(nd::Array::zeros([2]))
    .sample(
        distribution::multivar::gaussian(
            na::vector![0.5, 0.5],
            na::matrix![
                    0.01, 0.006;
                    0.006, 0.02;
            ],
        ),
    )
    .burn(1000)  // burn-in
```

