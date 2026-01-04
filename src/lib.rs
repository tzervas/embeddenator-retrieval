//! # embeddenator-retrieval
//!
//! Signature-based retrieval and resonator for holographic engrams.
//!
//! Extracted from embeddenator core as part of Phase 2A component decomposition.
//! See [ADR-016](https://github.com/tzervas/embeddenator/blob/main/docs/adr/ADR-016-component-decomposition.md).

pub mod retrieval;
pub mod core;

// Re-export key types
pub use retrieval::*;
pub use core::resonator;

#[cfg(test)]
mod tests {
    #[test]
    fn component_loads() {
        assert!(true);
    }
}
pub use core::correction;
