//! # embeddenator-retrieval
//!
//! Signature-based retrieval and resonator for holographic engrams.
//!
//! Extracted from embeddenator core as part of Phase 2A component decomposition.
//! See [ADR-016](https://github.com/tzervas/embeddenator/blob/main/docs/adr/ADR-016-component-decomposition.md).

pub mod core;
pub mod retrieval;

// Re-export key types
pub use core::correction;
pub use core::resonator;
pub use retrieval::*;

#[cfg(test)]
mod tests {
    #[test]
    fn component_loads() {
        // Basic sanity check that the crate compiles and loads
    }
}
