//! Unit tests for Resonator pattern completion and factorization

use embeddenator_retrieval::resonator::Resonator;
use embeddenator_vsa::{ReversibleVSAConfig, SparseVec};

fn enc(data: &[u8]) -> SparseVec {
    SparseVec::encode_data(data, &ReversibleVSAConfig::default(), None)
}

#[test]
fn test_resonator_new() {
    let resonator = Resonator::new();
    assert_eq!(resonator.max_iterations, 10);
    assert_eq!(resonator.convergence_threshold, 0.001);
    assert!(resonator.codebook.is_empty());
}

#[test]
fn test_resonator_with_params() {
    let codebook = vec![enc(b"pattern1"), enc(b"pattern2")];
    let resonator = Resonator::with_params(codebook.clone(), 20, 0.0001);
    assert_eq!(resonator.max_iterations, 20);
    assert_eq!(resonator.convergence_threshold, 0.0001);
    assert_eq!(resonator.codebook.len(), 2);
}

#[test]
fn test_resonator_project_clean_input() {
    let clean = enc(b"hello");
    let codebook = vec![clean.clone(), enc(b"world")];
    let resonator = Resonator::with_params(codebook, 10, 0.001);

    // Clean input should project to itself
    let projected = resonator.project(&clean);
    let similarity = clean.cosine(&projected);
    assert!(similarity > 0.9, "Similarity was {}", similarity);
}

#[test]
fn test_resonator_project_empty_codebook() {
    let resonator = Resonator::new();
    let input = enc(b"test");

    // Empty codebook should return input unchanged
    let projected = resonator.project(&input);
    assert_eq!(input.pos, projected.pos);
    assert_eq!(input.neg, projected.neg);
}

#[test]
fn test_resonator_factorize_empty_codebook() {
    let resonator = Resonator::new();
    let compound = enc(b"test");

    let result = resonator.factorize(&compound, 2);
    assert!(result.factors.is_empty());
    assert_eq!(result.iterations, 0);
    assert_eq!(result.final_delta, 0.0);
}

#[test]
fn test_resonator_factorize_zero_factors() {
    let codebook = vec![enc(b"pattern1")];
    let resonator = Resonator::with_params(codebook, 10, 0.001);
    let compound = enc(b"test");

    let result = resonator.factorize(&compound, 0);
    assert!(result.factors.is_empty());
    assert_eq!(result.iterations, 0);
    assert_eq!(result.final_delta, 0.0);
}

#[test]
fn test_resonator_factorize_convergence() {
    let factor1 = enc(b"hello");
    let factor2 = enc(b"world");
    let compound = factor1.bundle(&factor2);

    let codebook = vec![factor1.clone(), factor2.clone()];
    let resonator = Resonator::with_params(codebook, 20, 0.001);

    let result = resonator.factorize(&compound, 2);

    // Should return 2 factors
    assert_eq!(result.factors.len(), 2);
    // Should converge within reasonable iterations
    assert!(result.iterations <= 20);
    // Final delta should be reasonable
    assert!(result.final_delta >= 0.0);
    assert!(result.final_delta < 1.0);
}

#[test]
fn test_resonator_sign_threshold() {
    let resonator = Resonator::new();
    let similarities = vec![0.8, -0.3, 0.05, -0.9, 0.0];
    let ternary = resonator.sign_threshold(&similarities, 0.1);

    assert_eq!(ternary, vec![1, -1, 0, -1, 0]);
}

#[test]
fn test_resonator_sign_threshold_zero_threshold() {
    let resonator = Resonator::new();
    let similarities = vec![0.1, -0.1, 0.0];
    let ternary = resonator.sign_threshold(&similarities, 0.0);

    // With zero threshold, all non-zero values should be thresholded
    assert_eq!(ternary, vec![1, -1, 0]);
}

#[test]
fn test_resonator_sign_threshold_high_threshold() {
    let resonator = Resonator::new();
    let similarities = vec![0.5, -0.5, 0.05];
    let ternary = resonator.sign_threshold(&similarities, 0.6);

    // With high threshold, only strong similarities should pass
    assert_eq!(ternary, vec![0, 0, 0]);
}
