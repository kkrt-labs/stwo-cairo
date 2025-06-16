use std::simd::Simd;

use cairo_air::components::memory_id_to_big::{Claim, InteractionClaim, MEMORY_ID_SIZE};
use cairo_air::relations;
use itertools::{chain, izip, multiunzip, Itertools};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use stwo_cairo_adapter::memory::{
    u128_to_4_limbs, EncodedMemoryValueId, Memory, MemoryValueId, LARGE_MEMORY_VALUE_ID_BASE,
    RELOCATABLE_ID_BASE,
};
use stwo_cairo_common::memory::{
    MEMORY_ADDRESS_BOUND, N_M31_IN_FELT252, N_M31_IN_RELOCATABLE_FELT252, N_M31_IN_SMALL_FELT252,
};
use stwo_cairo_common::prover_types::felt::split_f252_simd;
use stwo_cairo_common::prover_types::simd::PackedFelt252;

use crate::witness::components::{
    range_check_9_9, range_check_9_9_b, range_check_9_9_c, range_check_9_9_d, range_check_9_9_e,
    range_check_9_9_f, range_check_9_9_g, range_check_9_9_h,
};
use crate::witness::prelude::*;
use crate::witness::utils::{AtomicMultiplicityColumn, TreeBuilder};

pub type InputType = M31;
pub type PackedInputType = PackedM31;

/// Generates the trace and the claim for the id -> f252 memory table.
/// Generates 2 table, one for large values and one for small values. A large value is a full 28
/// limb Felt252. The small values are currently 8 limbs, for a maximum of 72 bits.
/// The separation is done to reduce zeroed out ('unused') trace cells.
pub struct ClaimGenerator {
    big_values: Vec<[u32; 8]>,
    big_mults: AtomicMultiplicityColumn,
    small_values: Vec<u128>,
    small_mults: AtomicMultiplicityColumn,
    relocatable_values: Vec<[u32; 2]>,
    relocatable_mults: AtomicMultiplicityColumn,
}
impl ClaimGenerator {
    pub fn new(mem: &Memory) -> Self {
        // TODO(Ohad): pad to a power of 2 after splitting.
        let mut big_values = mem.f252_values.clone();
        let simd_padded_big_size = big_values.len().next_multiple_of(N_LANES);
        big_values.resize(simd_padded_big_size, [0; 8]);
        let big_mults = AtomicMultiplicityColumn::new(simd_padded_big_size);

        let mut small_values = mem.small_values.clone();
        let simd_padded_small_size = small_values.len().next_multiple_of(N_LANES);
        small_values.resize(simd_padded_small_size, 0);
        let small_mults = AtomicMultiplicityColumn::new(simd_padded_small_size);

        let mut relocatable_values = mem.relocatable_values.clone();
        let simd_padded_relocatable_size = relocatable_values.len().next_multiple_of(N_LANES);
        relocatable_values.resize(simd_padded_relocatable_size, [0; 2]);
        let relocatable_mults = AtomicMultiplicityColumn::new(simd_padded_relocatable_size);

        let big_size = big_values.len();
        let small_size = small_values.len();
        let relocatable_size = relocatable_values.len();

        assert!(
            big_size + small_size + relocatable_size <= MEMORY_ADDRESS_BOUND,
            "Assertion failed, condition `big_size ({big_size}) + small_size ({small_size}) + relocatable_size ({relocatable_size}) <= \
            MEMORY_ADDRESS_BOUND ({MEMORY_ADDRESS_BOUND})` is not satisfied."
        );

        Self {
            small_values,
            big_values,
            small_mults,
            big_mults,
            relocatable_values,
            relocatable_mults,
        }
    }

    pub fn deduce_output(&self, ids: PackedM31) -> PackedFelt252 {
        let values = std::array::from_fn(|j| {
            Simd::from_array(
                ids.to_array()
                    .map(|M31(i)| match EncodedMemoryValueId(i).decode() {
                        MemoryValueId::F252(id) => self.big_values[id as usize][j],
                        MemoryValueId::Small(id) => {
                            if j >= 4 {
                                0
                            } else {
                                let small = self.small_values[id as usize];
                                u128_to_4_limbs(small)[j]
                            }
                        }
                        MemoryValueId::MemoryRelocatable(id) => {
                            if j >= 2 {
                                0
                            } else {
                                self.relocatable_values[id as usize][1 - j]
                            }
                        }
                        MemoryValueId::Empty => {
                            panic!("Attempted deduce_output on empty memory cell.")
                        }
                    }),
            )
        });

        PackedFelt252 {
            value: split_f252_simd(values),
        }
    }

    pub fn add_inputs(&self, inputs: &[InputType]) {
        for input in inputs {
            self.add_input(input);
        }
    }

    pub fn add_packed_inputs(&self, inputs: &[PackedInputType]) {
        inputs.into_par_iter().for_each(|input| {
            self.add_packed_m31(input);
        });
    }

    pub fn add_packed_m31(&self, inputs: &PackedM31) {
        let memory_ids = inputs.to_array();
        for memory_id in memory_ids {
            self.add_input(&memory_id);
        }
    }

    pub fn add_input(&self, encoded_memory_id: &M31) {
        match EncodedMemoryValueId(encoded_memory_id.0).decode() {
            MemoryValueId::F252(id) => {
                self.big_mults.increase_at(id);
            }
            MemoryValueId::Small(id) => {
                self.small_mults.increase_at(id);
            }
            MemoryValueId::MemoryRelocatable(id) => {
                self.relocatable_mults.increase_at(id);
            }
            MemoryValueId::Empty => panic!("Attempted add_input on empty memory cell."),
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        range_check_9_9_trace_generator: &range_check_9_9::ClaimGenerator,
        range_check_9_9_b_trace_generator: &range_check_9_9_b::ClaimGenerator,
        range_check_9_9_c_trace_generator: &range_check_9_9_c::ClaimGenerator,
        range_check_9_9_d_trace_generator: &range_check_9_9_d::ClaimGenerator,
        range_check_9_9_e_trace_generator: &range_check_9_9_e::ClaimGenerator,
        range_check_9_9_f_trace_generator: &range_check_9_9_f::ClaimGenerator,
        range_check_9_9_g_trace_generator: &range_check_9_9_g::ClaimGenerator,
        range_check_9_9_h_trace_generator: &range_check_9_9_h::ClaimGenerator,
        log_max_big_size: u32,
    ) -> (Claim, InteractionClaimGenerator) {
        let big_table_traces = gen_big_memory_traces(
            self.big_values,
            self.big_mults.into_simd_vec(),
            log_max_big_size,
        );
        let small_table_trace =
            gen_small_memory_trace(self.small_values, self.small_mults.into_simd_vec());
        let relocatable_table_trace = gen_relocatable_memory_trace(
            &self.relocatable_values,
            &self.relocatable_mults.into_simd_vec(),
        );

        // Lookup data.
        let big_components_values: Vec<[_; N_M31_IN_FELT252]> = big_table_traces
            .iter()
            .map(|trace| std::array::from_fn(|i| trace[i].data.clone()))
            .collect_vec();
        let big_ids: Vec<_> = big_table_traces
            .iter()
            .map(|trace| trace[N_M31_IN_FELT252].data.clone())
            .collect_vec();
        let big_multiplicities: Vec<_> = big_table_traces
            .iter()
            .map(|trace| trace[N_M31_IN_FELT252 + 1].data.clone())
            .collect_vec();
        let small_values: [_; N_M31_IN_SMALL_FELT252] =
            std::array::from_fn(|i| small_table_trace[i].data.clone());
        let small_ids = small_table_trace[N_M31_IN_SMALL_FELT252].data.clone();
        let small_multiplicities = small_table_trace[N_M31_IN_SMALL_FELT252 + 1].data.clone();
        let relocatable_values: [_; N_M31_IN_RELOCATABLE_FELT252] =
            std::array::from_fn(|i| relocatable_table_trace[i].data.clone());
        let relocatable_ids = relocatable_table_trace[N_M31_IN_RELOCATABLE_FELT252].data.clone();
        let relocatable_multiplicities = relocatable_table_trace[N_M31_IN_RELOCATABLE_FELT252 + 1].data.clone();

        // Add inputs to range check that all the values are 9-bit felts.
        for values in &big_components_values {
            for (i, (col0, col1)) in values.iter().tuples().enumerate() {
                match i % 8 {
                    0 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    1 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_b_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    2 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_c_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    3 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_d_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    4 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_e_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    5 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_f_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    6 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_g_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    7 => col0
                        .par_iter()
                        .zip(col1.par_iter())
                        .for_each(|(val0, val1)| {
                            range_check_9_9_h_trace_generator.add_packed_m31(&[*val0, *val1]);
                        }),
                    _ => {
                        unreachable!("There are only 8 possible values for i % 8.",)
                    }
                };
            }
        }

        for (i, (col0, col1)) in small_values.iter().tuples().enumerate() {
            match i % 4 {
                0 => col0
                    .par_iter()
                    .zip(col1.par_iter())
                    .for_each(|(val0, val1)| {
                        range_check_9_9_trace_generator.add_packed_m31(&[*val0, *val1]);
                    }),
                1 => col0
                    .par_iter()
                    .zip(col1.par_iter())
                    .for_each(|(val0, val1)| {
                        range_check_9_9_b_trace_generator.add_packed_m31(&[*val0, *val1]);
                    }),
                2 => col0
                    .par_iter()
                    .zip(col1.par_iter())
                    .for_each(|(val0, val1)| {
                        range_check_9_9_c_trace_generator.add_packed_m31(&[*val0, *val1]);
                    }),
                3 => col0
                    .par_iter()
                    .zip(col1.par_iter())
                    .for_each(|(val0, val1)| {
                        range_check_9_9_d_trace_generator.add_packed_m31(&[*val0, *val1]);
                    }),
                _ => {
                    unreachable!("There are only 4 possible values for i % 4.",)
                }
            };
        }

        for (col0, col1) in relocatable_values.iter().tuples() {
            col0.par_iter()
                .zip(col1.par_iter())
                .for_each(|(val0, val1)| {
                    range_check_9_9_trace_generator.add_packed_m31(&[*val0, *val1]);
                });
        }

        // Extend trace.
        let mut big_log_sizes = vec![];
        for big_table_trace in big_table_traces {
            let big_log_size = big_table_trace[0].length.ilog2();
            big_log_sizes.push(big_log_size);
            let trace = big_table_trace
                .into_iter()
                .map(|eval| {
                    CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
                        CanonicCoset::new(big_log_size).circle_domain(),
                        eval,
                    )
                })
                .collect_vec();
            tree_builder.extend_evals(trace);
        }
        let small_log_size = small_table_trace[0].len().ilog2();
        let trace = small_table_trace
            .into_iter()
            .map(|eval| {
                CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
                    CanonicCoset::new(small_log_size).circle_domain(),
                    eval,
                )
            })
            .collect_vec();
        tree_builder.extend_evals(trace);
        let relocatable_log_size = relocatable_table_trace[0].len().ilog2();
        let trace = relocatable_table_trace
            .into_iter()
            .map(|eval| {
                CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
                    CanonicCoset::new(relocatable_log_size).circle_domain(),
                    eval,
                )
            })
            .collect_vec();
        tree_builder.extend_evals(trace);

        (
            Claim {
                big_log_sizes,
                small_log_size,
                relocatable_log_size,
            },
            InteractionClaimGenerator {
                big_components_values,
                big_ids,
                big_multiplicities,
                small_values,
                small_ids,
                small_multiplicities,
                relocatable_values,
                relocatable_ids,
                relocatable_multiplicities,
            },
        )
    }
}

/// Generates the trace for the id -> f252 `big` tables. Splits the table to multiple traces
/// according to `log_max_big_size`.
fn gen_big_memory_traces(
    values: Vec<[u32; 8]>,
    mults: Vec<PackedM31>,
    log_max_big_size: u32,
) -> Vec<Vec<BaseColumn>> {
    assert!(log_max_big_size >= LOG_N_LANES);
    let max_big_size = 1 << log_max_big_size;
    assert_eq!(values.len() / N_LANES, mults.len());
    let mut traces = vec![];

    for (values, mults) in values
        .chunks(max_big_size)
        .zip(mults.chunks(max_big_size / N_LANES))
    {
        let trace = gen_single_big_memory_trace(values, mults);
        traces.push(trace);
    }

    traces
}

// Generates the trace of the large value memory table.
fn gen_single_big_memory_trace(values: &[[u32; 8]], mults: &[PackedM31]) -> Vec<BaseColumn> {
    let column_length = mults
    .iter()
    .filter(|m| !m.is_zero())
    .count()
    .next_power_of_two();

    let (packed_values, mut ids, mut multiplicities): (
        Vec<[Simd<u32, N_LANES>; 8]>,
        Vec<PackedM31>,
        Vec<PackedM31>,
    ) = multiunzip(
        izip!(
            values
            .iter()
            .chain(std::iter::repeat(&[0; 8]))
            .take(column_length * N_LANES)
            .array_chunks::<N_LANES>(),
            (0..(column_length * N_LANES) as u32).array_chunks::<N_LANES>(),
            mults.iter()
        )
        .filter_map(|(v, i, m)| if m.is_zero() { None } else { Some((v, i, m)) })
        .map(|(v, i, m)| {
            (
                std::array::from_fn(|x| Simd::from_array(std::array::from_fn(|y| v[y][x]))),
                unsafe {
                    PackedM31::from_simd_unchecked(Simd::from_array(std::array::from_fn(|x| i[x])))
                },
                m,
            )
        }),
    );

    let mut values_trace = std::iter::repeat_with(|| BaseColumn::zeros(column_length))
        .take(N_M31_IN_FELT252)
        .collect_vec();
    for (i, values) in packed_values.iter().enumerate() {
        let values = split_f252_simd(*values);
        for (j, value) in values.iter().enumerate() {
            values_trace[j].data[i] = *value;
        }
    }
    ids.resize(column_length, PackedM31::zero());
    multiplicities.resize(column_length, PackedM31::zero());

    chain!(
        values_trace,
        [BaseColumn::from_simd(ids)],
        [BaseColumn::from_simd(multiplicities)]
    )
    .collect_vec()
}

// Generates the trace of the small value memory table.
fn gen_small_memory_trace(values: Vec<u128>, mults: Vec<PackedM31>) -> Vec<BaseColumn> {
    let column_length = mults
        .iter()
        .filter(|m| !m.is_zero())
        .count()
        .next_power_of_two();

    let values_len = values.len() as u32;

    let (packed_values, mut ids, mut multiplicities): (
        Vec<[Simd<u32, N_LANES>; 4]>,
        Vec<PackedM31>,
        Vec<PackedM31>,
    ) = multiunzip(
        izip!(
            values
                .into_iter()
                .map(u128_to_4_limbs)
                .array_chunks::<N_LANES>(),
            (0..values_len).array_chunks::<N_LANES>(),
            mults.iter()
        )
        .filter_map(|(v, i, m)| if m.is_zero() { None } else { Some((v, i, m)) })
        .map(|(v, i, m)| {
            (
                std::array::from_fn(|x| Simd::from_array(std::array::from_fn(|y| v[y][x]))),
                unsafe {
                    PackedM31::from_simd_unchecked(Simd::from_array(std::array::from_fn(|x| i[x])))
                },
                m,
            )
        }),
    );

    let mut values_trace = std::iter::repeat_with(|| BaseColumn::zeros(column_length * N_LANES))
        .take(N_M31_IN_SMALL_FELT252)
        .collect_vec();
    for (i, values) in packed_values.iter().enumerate() {
        let values = split_f252_simd([
            values[0],
            values[1],
            values[2],
            values[3],
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
        ]);
        for (j, value) in values[..N_M31_IN_SMALL_FELT252].iter().enumerate() {
            values_trace[j].data[i] = *value;
        }
    }

    ids.resize(column_length, PackedM31::zero());
    multiplicities.resize(column_length, PackedM31::zero());

    chain!(
        values_trace,
        [BaseColumn::from_simd(ids)],
        [BaseColumn::from_simd(multiplicities)]
    )
    .collect_vec()
}

fn gen_relocatable_memory_trace(values: &Vec<[u32; 2]>, mults: &Vec<PackedM31>) -> Vec<BaseColumn> {
    let column_length = mults
        .iter()
        .filter(|m| !m.is_zero())
        .count()
        .next_power_of_two();

    let values_len = values.len() as u32;

    let (packed_values, mut ids, mut multiplicities): (
        Vec<[Simd<u32, N_LANES>; 2]>,
        Vec<PackedM31>,
        Vec<PackedM31>,
    ) = multiunzip(
        izip!(
            values
                .into_iter()
                .array_chunks::<N_LANES>(),
            (0..values_len).array_chunks::<N_LANES>(),
            mults.iter()
        )
        .filter_map(|(v, i, m)| if m.is_zero() { None } else { Some((v, i, m)) })
        .map(|(v, i, m)| {
            (
                std::array::from_fn(|x| Simd::from_array(std::array::from_fn(|y| v[y][x]))),
                unsafe {
                    PackedM31::from_simd_unchecked(Simd::from_array(std::array::from_fn(|x| i[x])))
                },
                m,
            )
        }),
    );

    let mut values_trace = std::iter::repeat_with(|| BaseColumn::zeros(column_length * N_LANES))
        .take(N_M31_IN_RELOCATABLE_FELT252)
        .collect_vec();

    for (i, values) in packed_values.iter().enumerate() {
        let values = split_f252_simd([
            values[1],
            values[0],
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
        ]);
        for (j, value) in values[..N_M31_IN_RELOCATABLE_FELT252].iter().enumerate() {
            values_trace[j].data[i] = *value;
        }
    }

    ids.resize(column_length, PackedM31::zero());
    multiplicities.resize(column_length, PackedM31::zero());

    chain!(
        values_trace,
        [BaseColumn::from_simd(ids)],
        [BaseColumn::from_simd(multiplicities)]
    )
    .collect_vec()
}

#[derive(Debug)]
pub struct InteractionClaimGenerator {
    pub big_components_values: Vec<[Vec<PackedM31>; N_M31_IN_FELT252]>,
    pub big_ids: Vec<Vec<PackedM31>>,
    pub big_multiplicities: Vec<Vec<PackedM31>>,
    pub small_values: [Vec<PackedM31>; N_M31_IN_SMALL_FELT252],
    pub small_ids: Vec<PackedM31>,
    pub small_multiplicities: Vec<PackedM31>,
    pub relocatable_values: [Vec<PackedM31>; N_M31_IN_RELOCATABLE_FELT252],
    pub relocatable_ids: Vec<PackedM31>,
    pub relocatable_multiplicities: Vec<PackedM31>,
}
impl InteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
        range9_9_b_lookup_elements: &relations::RangeCheck_9_9_B,
        range9_9_c_lookup_elements: &relations::RangeCheck_9_9_C,
        range9_9_d_lookup_elements: &relations::RangeCheck_9_9_D,
        range9_9_e_lookup_elements: &relations::RangeCheck_9_9_E,
        range9_9_f_lookup_elements: &relations::RangeCheck_9_9_F,
        range9_9_g_lookup_elements: &relations::RangeCheck_9_9_G,
        range9_9_h_lookup_elements: &relations::RangeCheck_9_9_H,
    ) -> InteractionClaim {
        let mut offset = 0;
        let (big_traces, big_claimed_sums): (Vec<_>, Vec<_>) = 
            izip!(
                self.big_components_values.iter(),
                self.big_multiplicities.iter(),
                self.big_ids.iter(),
            )
            .map(|(big_components_values, big_multiplicities, big_ids)| {
                let res = Self::gen_big_memory_interaction_trace(
                    big_components_values,
                    big_ids,
                    big_multiplicities,
                    offset,
                    lookup_elements,
                    range9_9_lookup_elements,
                    range9_9_b_lookup_elements,
                    range9_9_c_lookup_elements,
                    range9_9_d_lookup_elements,
                    range9_9_e_lookup_elements,
                    range9_9_f_lookup_elements,
                    range9_9_g_lookup_elements,
                    range9_9_h_lookup_elements,
                );
                offset += big_multiplicities.len() as u32 * N_LANES as u32;
                res
            })
            .unzip();
        for big_trace in big_traces {
            tree_builder.extend_evals(big_trace);
        }

        let (small_trace, small_claimed_sum) = self.gen_small_memory_interaction_trace(
            lookup_elements,
            range9_9_lookup_elements,
            range9_9_b_lookup_elements,
            range9_9_c_lookup_elements,
            range9_9_d_lookup_elements,
        );
        tree_builder.extend_evals(small_trace);

        let (relocatable_trace, relocatable_claimed_sum) = self
            .gen_relocatable_memory_interaction_trace(lookup_elements, range9_9_lookup_elements);
        tree_builder.extend_evals(relocatable_trace);

        InteractionClaim {
            small_claimed_sum,
            big_claimed_sums,
            relocatable_claimed_sum,
        }
    }

    fn gen_big_memory_interaction_trace(
        big_components_values: &[Vec<PackedM31>; N_M31_IN_FELT252],
        big_ids: &[PackedM31],
        big_multiplicities: &[PackedM31],
        offset: u32,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
        range9_9_b_lookup_elements: &relations::RangeCheck_9_9_B,
        range9_9_c_lookup_elements: &relations::RangeCheck_9_9_C,
        range9_9_d_lookup_elements: &relations::RangeCheck_9_9_D,
        range9_9_e_lookup_elements: &relations::RangeCheck_9_9_E,
        range9_9_f_lookup_elements: &relations::RangeCheck_9_9_F,
        range9_9_g_lookup_elements: &relations::RangeCheck_9_9_G,
        range9_9_h_lookup_elements: &relations::RangeCheck_9_9_H,
    ) -> (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        QM31,
    ) {
        assert!(big_components_values
            .iter()
            .all(|v| v.len() == big_multiplicities.len()));
        let big_table_log_size = big_components_values[0].len().ilog2() + LOG_N_LANES;
        let mut big_values_logup_gen = LogupTraceGenerator::new(big_table_log_size);

        // Every element is 9-bit.
        for (i, (limb0, limb1, limb2, limb3)) in big_components_values.iter().tuples().enumerate() {
            let mut col_gen = big_values_logup_gen.new_col();
            (col_gen.par_iter_mut(), limb0, limb1, limb2, limb3)
                .into_par_iter()
                .for_each(|(writer, limb0, limb1, limb2, limb3)| {
                    let (denom0, denom1): (PackedQM31, PackedQM31) = match i % 4 {
                        0 => (
                            range9_9_lookup_elements.combine(&[*limb0, *limb1]),
                            range9_9_b_lookup_elements.combine(&[*limb2, *limb3]),
                        ),
                        1 => (
                            range9_9_c_lookup_elements.combine(&[*limb0, *limb1]),
                            range9_9_d_lookup_elements.combine(&[*limb2, *limb3]),
                        ),
                        2 => (
                            range9_9_e_lookup_elements.combine(&[*limb0, *limb1]),
                            range9_9_f_lookup_elements.combine(&[*limb2, *limb3]),
                        ),
                        3 => (
                            range9_9_g_lookup_elements.combine(&[*limb0, *limb1]),
                            range9_9_h_lookup_elements.combine(&[*limb2, *limb3]),
                        ),
                        _ => {
                            unreachable!("There are only 4 possible values for i % 4.",)
                        }
                    };
                    writer.write_frac(denom0 + denom1, denom0 * denom1);
                });
            col_gen.finalize_col();
        }

        // Yield large values.
        let mut col_gen = big_values_logup_gen.new_col();
        let large_memory_value_id_tag = PackedM31::broadcast(M31::from_u32_unchecked(LARGE_MEMORY_VALUE_ID_BASE));
        let packed_offset = PackedM31::broadcast(M31::from_u32_unchecked(offset));
        for vec_row in 0..1 << (big_table_log_size - LOG_N_LANES) {
            let id_and_value: [_; N_M31_IN_FELT252 + MEMORY_ID_SIZE] = std::array::from_fn(|i| {
                if i == 0 {
                    big_ids[vec_row] + large_memory_value_id_tag + packed_offset
                } else {
                    big_components_values[i - 1][vec_row]
                }
            });
            let denom: PackedQM31 = lookup_elements.combine(&id_and_value);
            col_gen.write_frac(vec_row, (-big_multiplicities[vec_row]).into(), denom);
        }
        col_gen.finalize_col();

        big_values_logup_gen.finalize_last()
    }

    fn gen_small_memory_interaction_trace(
        &self,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
        range9_9_b_lookup_elements: &relations::RangeCheck_9_9_B,
        range9_9_c_lookup_elements: &relations::RangeCheck_9_9_C,
        range9_9_d_lookup_elements: &relations::RangeCheck_9_9_D,
    ) -> (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        QM31,
    ) {
        let small_table_log_size = self.small_values[0].len().ilog2() + LOG_N_LANES;
        let mut small_values_logup_gen = LogupTraceGenerator::new(small_table_log_size);

        // Every element is 9-bit.
        for (i, (l, r)) in self.small_values.iter().tuples().enumerate() {
            let mut col_gen = small_values_logup_gen.new_col();
            (col_gen.par_iter_mut(), l, r)
                .into_par_iter()
                .for_each(|(writer, l1, l2)| {
                    // TODO(alont) Add 2-batching.
                    let denom = match i % 4 {
                        0 => range9_9_lookup_elements.combine(&[*l1, *l2]),
                        1 => range9_9_b_lookup_elements.combine(&[*l1, *l2]),
                        2 => range9_9_c_lookup_elements.combine(&[*l1, *l2]),
                        3 => range9_9_d_lookup_elements.combine(&[*l1, *l2]),
                        _ => {
                            unreachable!("There are only 4 possible values for i % 4.",)
                        }
                    };
                    writer.write_frac(PackedQM31::broadcast(M31(1).into()), denom);
                });
            col_gen.finalize_col();
        }

        // Yield small values.
        let mut col_gen = small_values_logup_gen.new_col();
        for vec_row in 0..1 << (small_table_log_size - LOG_N_LANES) {
            let id_and_value: [_; N_M31_IN_SMALL_FELT252 + MEMORY_ID_SIZE] =
                std::array::from_fn(|i| {
                    if i == 0 {
                        self.small_ids[vec_row]
                    } else {
                        self.small_values[i - 1][vec_row]
                    }
                });
            let denom: PackedQM31 = lookup_elements.combine(&id_and_value);
            col_gen.write_frac(vec_row, (-self.small_multiplicities[vec_row]).into(), denom);
        }
        col_gen.finalize_col();

        small_values_logup_gen.finalize_last()
    }

    fn gen_relocatable_memory_interaction_trace(
        &self,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
    ) -> (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        QM31,
    ) {
        let relocatable_table_log_size = self.relocatable_values[0].len().ilog2() + LOG_N_LANES;
        let mut relocatable_values_logup_gen = LogupTraceGenerator::new(relocatable_table_log_size);

        // Every element is 9-bit.
        for (l, r) in self.relocatable_values.iter().tuples() {
            let mut col_gen = relocatable_values_logup_gen.new_col();
            (col_gen.par_iter_mut(), l, r)
                .into_par_iter()
                .for_each(|(writer, l1, l2)| {
                    // TOOD(alont) Add 2-batching.
                    writer.write_frac(
                        PackedQM31::broadcast(M31(1).into()),
                        range9_9_lookup_elements.combine(&[*l1, *l2]),
                    );
                });
            col_gen.finalize_col();
        }

        // Yield relocatable values.
        let mut col_gen = relocatable_values_logup_gen.new_col();
        let relocatable_memory_value_id_tag = PackedM31::broadcast(M31::from_u32_unchecked(RELOCATABLE_ID_BASE));
        for vec_row in 0..1 << (relocatable_table_log_size - LOG_N_LANES) {
            let id_and_value: [_; N_M31_IN_RELOCATABLE_FELT252 + MEMORY_ID_SIZE] =
                std::array::from_fn(|i| {
                    if i == 0 {
                        self.relocatable_ids[vec_row] + relocatable_memory_value_id_tag
                    } else {
                        self.relocatable_values[i - 1][vec_row]
                    }
                });
            let denom: PackedQM31 = lookup_elements.combine(&id_and_value);
            col_gen.write_frac(
                vec_row,
                (-self.relocatable_multiplicities[vec_row]).into(),
                denom,
            );
        }
        col_gen.finalize_col();

        relocatable_values_logup_gen.finalize_last()
    }
}
