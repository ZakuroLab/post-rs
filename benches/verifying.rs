use criterion::{criterion_group, criterion_main, Criterion};
use post::{
    metadata::ProofMetadata,
    prove::Proof,
    verification::{Verifier, VerifyingParams},
};
use pprof::criterion::{Output, PProfProfiler};

use scrypt_jane::scrypt::ScryptParams;

fn verifying(c: &mut Criterion) {
    let challenge = b"hello world, challenge me!!!!!!!";
    let metadata = ProofMetadata {
        node_id: [0u8; 32],
        commitment_atx_id: [0u8; 32],
        challenge: *challenge,
        num_units: 1,
        labels_per_unit: 1024 * 1024 * 1024,
    };
    let num_labels = metadata.num_units as u64 * metadata.labels_per_unit;

    let verifier = Verifier::new(post::pow::RandomXFlag::get_recommended_flags()).unwrap();

    let (k2, k3) = (37, 37);
    let proof = Proof::new(
        0,
        (0..k2 as u64).collect::<Vec<u64>>().as_slice(),
        num_labels.ilog2() as usize + 1,
        0,
    );
    let params = VerifyingParams {
        difficulty: u64::MAX,
        k2,
        k3,
        pow_difficulty: [0xFF; 32],
        scrypt: ScryptParams::new(12, 0, 0),
    };

    c.bench_function("verify", |b| {
        b.iter(|| {
            verifier
                .verify(&proof, &metadata, params)
                .expect("proof should be valid");
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(1000, Output::Flamegraph(None)));
    targets=verifying
);

criterion_main!(benches);
