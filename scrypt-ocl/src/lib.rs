use ocl::DeviceType;
use post::{
    initialize::{Initialize, VrfNonce, ENTIRE_LABEL_SIZE, LABEL_SIZE},
};
use std::{cmp::min, fmt::Display, io::Write, ops::Range};
use thiserror::Error;

pub use ocl;

use spacemesh_cuda;

mod filtering;

#[derive(Debug)]
struct Scrypter {
    labels_buffer: Vec<u8>,
    N: u32,
    device_id: u32,
}

#[derive(Error, Debug)]
pub enum ScryptError {
    #[error("Labels range too big to fit in usize")]
    LabelsRangeTooBig,
    #[error("Invalid buffer size: got {got}, expected {expected}")]
    InvalidBufferSize { got: usize, expected: usize },
    #[error("Fail in OpenCL: {0}")]
    OclError(#[from] ocl::Error),
    #[error("Fail in OpenCL core: {0}")]
    OclCoreError(#[from] ocl::OclCoreError),
    #[error("Invalid provider id: {0:?}")]
    InvalidProviderId(ProviderId),
    #[error("No providers available")]
    NoProvidersAvailable,
    #[error("Failed to write labels: {0}")]
    WriteError(#[from] std::io::Error),
}

macro_rules! cast {
    ($target: expr, $pat: path) => {{
        if let $pat(a) = $target {
            // #1
            a
        } else {
            panic!("mismatch variant when cast to {}", stringify!($pat)); // #2
        }
    }};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderId(pub u32);

pub struct Provider {
    pub class: DeviceType,
    pub device_id: u32,
}

impl Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] device_id: {}", self.class, self.device_id)
    }
}

pub fn get_providers_count(device_types: Option<DeviceType>) -> usize {
    get_providers(device_types).map_or_else(
        |e| {
            log::error!("failed to get providers: {e}");
            0
        },
        |p| p.len(),
    )
}

pub fn get_providers(device_types: Option<DeviceType>) -> Result<Vec<Provider>, ScryptError> {
    let device_types = device_types.or(Some(DeviceType::GPU)).unwrap();

    if !device_types.contains(DeviceType::GPU) {
        panic!("only support gpu device type");
    }

    let device_num = spacemesh_cuda::get_device_num();

    let mut providers = Vec::new();

    for t in 0..device_num {
        providers.push(Provider {
            class: DeviceType::GPU,
            device_id: t,
        })
    }
    Ok(providers)
}

fn scan_for_vrf_nonce(labels: &[u8], mut difficulty: [u8; 32]) -> Option<VrfNonce> {
    let mut nonce = None;
    for (id, label) in labels.chunks(ENTIRE_LABEL_SIZE).enumerate() {
        if label < &difficulty {
            nonce = Some(VrfNonce {
                index: id as u64,
                label: label.try_into().unwrap(),
            });
            difficulty = label.try_into().unwrap();
        }
    }
    nonce
}

impl Scrypter {
    pub fn new(n: usize, device_id: u32) -> Result<Self, ScryptError> {
        // n = 8192
        let max_task_num = spacemesh_cuda::get_max_task_num(device_id) as usize;
        Ok(Self {
            labels_buffer: vec![0u8; max_task_num * ENTIRE_LABEL_SIZE],
            N: n as u32,
            device_id,
        })
    }

    pub fn scrypt<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        labels: Range<u64>,
        commitment: &[u8; 32],
        mut vrf_difficulty: Option<[u8; 32]>,
    ) -> Result<Option<VrfNonce>, ScryptError> {
        let commitment: Vec<u32> = commitment
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // self.input.write(&commitment).enq()?;

        let max_task_num = spacemesh_cuda::get_max_task_num(self.device_id) as usize;

        let mut best_nonce = None;
        let labels_end = labels.end;
        let label_num = labels.end - labels.start;
        if label_num % 32 != 0 {
            panic!("the number of labels must be a multiple of 32");
        }

        for index in labels.step_by(max_task_num) {
            let index_end = min(index + max_task_num as u64, labels_end);
            let labels_to_init = (index_end - index) as usize;
            spacemesh_cuda::scrypt(
                self.device_id,
                index,
                commitment.as_ref(),
                labels_to_init as u32,
                self.labels_buffer.as_mut(),
            ).expect("Failed to execute the scrypt process");

            let labels_buffer =
                &mut self.labels_buffer.as_mut_slice()[..labels_to_init * ENTIRE_LABEL_SIZE];

            // Look for VRF nonce if enabled
            // TODO: run in background / in parallel to GPU
            if let Some(difficulty) = vrf_difficulty {
                if let Some(nonce) = scan_for_vrf_nonce(labels_buffer, difficulty) {
                    best_nonce = Some(VrfNonce {
                        index: nonce.index + index,
                        label: nonce.label,
                    });
                    vrf_difficulty = Some(nonce.label);
                    log::trace!("Found new smallest nonce: {best_nonce:?}");
                }
            }

            // Move labels in labels_buffer, taking only 16B of each label in-place, creating a continuous buffer of 16B labels.
            // TODO: run in background / in parallel to GPU
            let mut dst = 0;
            for label_id in 0..labels_to_init {
                let src = label_id * ENTIRE_LABEL_SIZE;
                labels_buffer.copy_within(src..src + LABEL_SIZE, dst);
                dst += LABEL_SIZE;
            }
            writer.write_all(&labels_buffer[..dst])?;
        }
        Ok(best_nonce)
    }
}

pub struct OpenClInitializer {
    scrypter: Scrypter,
}

impl OpenClInitializer {
    pub fn new(
        provider_id: Option<ProviderId>,
        n: usize,
        device_types: Option<DeviceType>,
    ) -> Result<Self, ScryptError> {
        let providers = get_providers(device_types)?;
        let provider = if let Some(id) = provider_id {
            log::info!(
                "selecting {} provider from {} available",
                id.0,
                providers.len()
            );
            providers
                .get(id.0 as usize)
                .ok_or(ScryptError::InvalidProviderId(id))?
        } else {
            providers.first().ok_or(ScryptError::NoProvidersAvailable)?
        };

        let device_id = provider.device_id;
        log::info!("Using provider: {provider}");

        let scrypter = Scrypter::new(n, device_id)?;

        Ok(Self { scrypter })
    }
}

impl Initialize for OpenClInitializer {
    fn initialize_to(
        &mut self,
        writer: &mut dyn Write,
        commitment: &[u8; 32],
        labels: Range<u64>,
        vrf_difficulty: Option<[u8; 32]>,
    ) -> Result<Option<VrfNonce>, Box<dyn std::error::Error>> {
        self.scrypter
            .scrypt(writer, labels, commitment, vrf_difficulty)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use post::{
        config::ScryptParams,
        initialize::{CpuInitializer, Initialize},
    };
    use rstest::rstest;

    use super::*;

    #[test]
    fn scanning_for_vrf_nonce() {
        let labels = [[0xFF; 32], [0xEE; 32], [0xDD; 32], [0xEE; 32]];
        let labels_bytes: Vec<u8> = labels.iter().copied().flatten().collect();
        let nonce = scan_for_vrf_nonce(&labels_bytes, [0xFFu8; 32]);
        assert_eq!(
            nonce,
            Some(VrfNonce {
                index: 2,
                label: [0xDD; 32]
            })
        );
    }

    #[test]
    fn scrypting_1_label() {
        let mut scrypter = OpenClInitializer::new(Some(ProviderId(0)), 8192, None).unwrap();
        let mut labels = Vec::new();
        scrypter
            .initialize_to(&mut labels, &[8u8; 32], 0..32 * 3, None)
            .unwrap();

        let mut expected = Vec::with_capacity(32);
        CpuInitializer::new(ScryptParams::new(8192, 1, 1))
            .initialize_to(&mut expected, &[8u8; 32], 0..32 * 3, None)
            .unwrap();

        assert_eq!(expected, labels);
    }

    #[rstest]
    #[case(8192)]
    fn scrypting_from_0(#[case] n: usize) {
        let indices = 0..(1024 * 64);

        let mut scrypter = OpenClInitializer::new(None, n, None).unwrap();
        let mut labels = Vec::new();
        scrypter
            .initialize_to(&mut labels, &[3u8; 32], indices.clone(), None)
            .unwrap();

        let mut expected =
            Vec::<u8>::with_capacity(usize::try_from(indices.end - indices.start).unwrap());

        CpuInitializer::new(ScryptParams::new(n, 1, 1))
            .initialize_to(&mut expected, &[3u8; 32], indices, None)
            .unwrap();

        assert_eq!(expected, labels);
    }

    #[rstest]
    #[case(8192)]
    fn scrypting_over_4gb(#[case] n: usize) {
        let indices = u32::MAX as u64 - 1000..u32::MAX as u64 + 1000;

        let mut scrypter = OpenClInitializer::new(None, n, None).unwrap();
        let mut labels = Vec::new();
        scrypter
            .initialize_to(&mut labels, &[0u8; 32], indices.clone(), None)
            .unwrap();

        let mut expected =
            Vec::<u8>::with_capacity(usize::try_from(indices.end - indices.start).unwrap());

        CpuInitializer::new(ScryptParams::new(n, 1, 1))
            .initialize_to(&mut expected, &[0u8; 32], indices, None)
            .unwrap();

        assert_eq!(expected, labels);
    }

    #[test]
    fn scrypting_with_commitment() {
        let indices = 11..(32 * 3) + 11;
        let commitment = b"this is some commitment for init";

        let mut scrypter = OpenClInitializer::new(None, 8192, None).unwrap();
        let mut labels = Vec::new();
        scrypter
            .initialize_to(&mut labels, commitment, indices.clone(), None)
            .unwrap();

        let mut expected =
            Vec::<u8>::with_capacity(usize::try_from(indices.end - indices.start).unwrap());

        CpuInitializer::new(ScryptParams::new(8192, 1, 1))
            .initialize_to(&mut expected, commitment, indices, None)
            .unwrap();

        assert_eq!(expected, labels);
    }

    #[rstest]
    #[case(8192)]
    fn searching_for_vrf_nonce(#[case] n: usize) {
        let indices = 0..6000;
        let commitment = b"this is some commitment for init";
        let mut difficulty = [0xFFu8; 32];
        difficulty[0] = 0;
        difficulty[1] = 0x2F;

        let mut scrypter = OpenClInitializer::new(None, n, None).unwrap();
        let mut labels = Vec::new();
        let opencl_nonce = scrypter
            .initialize_to(&mut labels, commitment, indices.clone(), Some(difficulty))
            .unwrap();
        let nonce = opencl_nonce.expect("vrf nonce not found");

        let mut label = Vec::<u8>::with_capacity(LABEL_SIZE);
        let mut cpu_initializer = CpuInitializer::new(ScryptParams::new(n, 1, 1));
        cpu_initializer
            .initialize_to(&mut label, commitment, nonce.index..nonce.index + 1, None)
            .unwrap();

        assert_eq!(&nonce.label[..16], label.as_slice());
        assert!(nonce.label.as_slice() < &difficulty);
        assert!(label.as_slice() < &difficulty);

        let mut sink = std::io::sink();
        let cpu_nonce = cpu_initializer
            .initialize_to(&mut sink, commitment, indices, Some(difficulty))
            .unwrap();

        assert_eq!(cpu_nonce, opencl_nonce);
    }
}
