# Probs

Draw samples in any distribution with various sampling techniques. 



## Usage

Construct a sampler with [builder pattern](https://doc.rust-lang.org/1.0.0/style/ownership/builders.html) 

```rust
use randvar::modular::Z;
use sampler::Sampler;
fn test() {
  sampler::univar::Icdf::<Z<256>>::new()
    .burn(100)  // burn-in period
    .pick(3)    // pick 3 samples
```

Sample a distribution with it 

```rust
    .sample(dist::univar::gaussian(128.0, 32.0))
```

Use it like any [Iterator](https://doc.rust-lang.org/book/ch13-02-iterators.html) as you would 

```rust
    .enumerate()
    .for_each(|(i, x)| println!("sample#{}: {}", i, x))
}
```



## Distribution

**ANY** customized/weird distribution can be sampled with provided samplers

```rust
sampler::univar::Icdf::<Z<256>>::new()
  .sample(|Z(x)| (x % 8) as f64)
```

> Some sampler needs a burn-in period to achieve equilibrium



## Multi-dimensional

[Gibbs Sampler](https://wikipedia.org/wiki/Gibbs_sampling) is used to sample distribution from high dimension

- Fix-dimensioned domain is represented with [ndarray](https://crates.io/crates/ndarray)

```rust
univar::Icdf::<Z<256>>::new()
    .gibbs(nd::Dim([2]))
    .burn(1000)
    .sample(dist::multivar::gaussian(
        na::vector![128.0, 128.0], // μ
        na::matrix![               // σ
            128.0, 32.0;
            32.0, 64.0;
        ],
    ))
```

