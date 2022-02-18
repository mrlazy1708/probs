# probs

Provide definition of `domain` and basic `distribution`. Implement basic `sampler` with `Iterator`.



## Basic Usage

Construct a sampler of certain sampling technique

```rust
use sampler::Global;
fn test() {
  sampler::univar::Icdf::new()
```

Sample a distribution with it

```rust
		.sample(distribution::univar::normal::<i8>(0.0, 32.0))
```

Use it like an `Iterator` as you would

```rust
		.enumerate()
		.for_each(|(i, x)| println!("sample#{}: {}", i, x))
}
```



## Arbitrary Distribution

**ANY** customized distribution can be sampled with provided samplers

```rust
sampler::univar::Icdf::new()
	.sample(|x: &u8| (x % 8) as f64)
```



## Multi-dimensional

[Gibbs Sampler](https://wikipedia.org/wiki/Gibbs_sampling) is used to sample distribution on high dimension

- Fix-dimensioned domain is represented with [ndarray](https://crates.io/crates/ndarray)

```rust
sampler::univar::Icdf::new()
	.gibbs(nd::Dim([2, 2]), 100) // 2x2 domain; skip first 100 samples
	.sample(|m: &nd::Array2<u8>| m.sum() as f64)
```

