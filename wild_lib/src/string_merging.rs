use crate::alignment;
use crate::debug_assert_bail;
use crate::error::Result;
use crate::hash::PassThroughHasher;
use crate::hash::PreHashed;
use crate::input_data::FileId;
use crate::output_section_id::OutputSectionId;
use crate::output_section_id::OutputSections;
use crate::output_section_map::OutputSectionMap;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::part_id;
use crate::part_id::PartId;
use crate::resolution::ResolvedFile;
use crate::resolution::ResolvedGroup;
use crate::resolution::SectionSlot;
use anyhow::bail;
use anyhow::Context;
use dashmap::DashMap;
use fxhash::FxHashMap;
use object::read::elf::Sym as _;
use object::LittleEndian;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

const MERGE_STRING_BUCKETS: usize = 32;

#[derive(Clone, Copy)]
pub(crate) struct MergeStringsFileSection {
    pub(crate) part_id: PartId,

    /// DO NOT SUBMIT: Document and possibly rename. Check all references. Maybe rename the struct
    /// to which this refers. Do we want a newtype here?
    pub(crate) merge_info_index: u32,
}

pub(crate) struct MergeStringSectionSizes {
    sizes: OutputSectionMap<u32>,
}

pub(crate) struct MergeStringsFileSectionData<'data> {
    pub(crate) input_section_index: object::SectionIndex,

    pub(crate) part_id: PartId,

    offset_to_id: FxHashMap<u64, MergedStringId>,

    pub(crate) section_data: &'data [u8],

    pub(crate) strings: Vec<MergeStringData>,

    /// Total size of all our strings.
    pub(crate) size: u64,
}

#[derive(Default)]
pub(crate) struct MergeStringsSection {
    // DO NOT SUBMIT: Is this offset or address?
    /// Mapping from MergedStringId to the offset of that string within the output section.
    pub(crate) id_to_offset: Vec<u64>,
}

#[derive(Clone, Copy)]
pub(crate) struct MergeStringData {
    id: MergedStringId,
    range: MergeStringRange,
}

#[derive(Clone, Copy)]
struct MergeStringRange {
    /// The offset of the start of this string within the input section.
    start_offset: u32,

    /// The exclusive end of this string within the input section. The string includes the null
    /// terminator, so this should point one byte past the null terminator.
    end_offset: u32,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(crate) struct StringToMerge<'data> {
    bytes: &'data [u8],
}

/// The addresses of the start of the merged strings for each output section.
pub(crate) struct MergedStringStartAddresses {
    addresses: OutputSectionMap<u64>,
}

#[derive(Default)]
struct MergeStringSectionBuilder<'data> {
    string_to_info: DashMap<PreHashed<StringToMerge<'data>>, MergeStringInfo, PassThroughHasher>,
    next_id: AtomicU32,
    input_byte_count: AtomicU64,
    input_string_count: AtomicU64,
}

#[derive(Clone, Copy)]
struct MergeStringInfo {
    id: MergedStringId,

    /// The file that is responsible for providing the definition of this string.
    file_id: FileId,

    /// The section of the input file that this string came from.
    section_index: object::SectionIndex,

    range: MergeStringRange,
}

/// An internal ID for a merged string. Within an output section, an ID uniquely identifies a
/// particular string. These IDs are allocated from multiple threads, so are non-deterministic and
/// should not be used in a way that will affect output.
#[derive(Clone, Copy)]
struct MergedStringId(u32);

/// Merges identical strings from all loaded objects where those strings are from input sections
/// that are marked with both the SHF_MERGE and SHF_STRINGS flags. The return value is allocated
/// space for filling in the addresses of each merged string.
#[tracing::instrument(skip_all, name = "Merge strings")]
pub(crate) fn merge_strings<'data>(
    resolved: &mut [ResolvedGroup<'data>],
    output_sections: &OutputSections,
) -> Result<MergeStringSectionSizes> {
    let merge_builders: OutputSectionMap<MergeStringSectionBuilder> =
        output_sections.new_section_map();

    deduplicate_merge_strings(resolved, &merge_builders)?;

    print_merge_metrics(&merge_builders, output_sections);

    let sizes = MergeStringSectionSizes::new(&merge_builders);

    partition_merged_string_records(resolved, merge_builders)?;

    sort_merged_string_records(resolved);

    Ok(sizes)
}

#[tracing::instrument(skip_all, name = "Split and deduplicate merge-strings")]
fn deduplicate_merge_strings<'data>(
    resolved: &mut [ResolvedGroup<'data>],
    merge_builders: &OutputSectionMap<MergeStringSectionBuilder<'data>>,
) -> Result {
    resolved.par_iter_mut().try_for_each(|group| {
        for file in &mut group.files {
            let ResolvedFile::Object(obj) = file else {
                continue;
            };
            let Some(non_dynamic) = obj.non_dynamic.as_mut() else {
                continue;
            };

            let file_id = obj.file_id;

            for merge_info in &mut non_dynamic.merge_strings_sections {
                let SectionSlot::MergeStrings(sec) =
                    non_dynamic.sections[merge_info.input_section_index.0]
                else {
                    bail!("Internal error: expected SectionSlot::MergeStrings");
                };

                merge_info.part_id = sec.part_id;
                let builder = merge_builders.get(sec.part_id.output_section_id());

                let mut remaining = merge_info.section_data;
                let mut offset = 0;

                while !remaining.is_empty() {
                    let string = StringToMerge::take_hashed(&mut remaining)?;

                    let string_id = builder.process_string(
                        string,
                        file_id,
                        merge_info.input_section_index,
                        MergeStringRange {
                            start_offset: offset as u32,
                            end_offset: offset as u32 + string.bytes.len() as u32,
                        },
                    );

                    merge_info.offset_to_id.insert(offset, string_id);
                    offset += string.bytes.len() as u64;
                }
            }
        }
        Ok(())
    })
}

/// Adds all merge-string records to the file-merge-section that is responsible for it.
#[tracing::instrument(skip_all, name = "Partition merged-string records")]
fn partition_merged_string_records<'data>(
    resolved: &mut [ResolvedGroup<'data>],
    merge_builders: OutputSectionMap<MergeStringSectionBuilder<'data>>,
) -> Result {
    // This may not be the optimal approach performance-wise. Experimentation with alternatives is
    // encouraged.
    let mut sections = HashMap::new();
    for group in resolved {
        for file in &mut group.files {
            let ResolvedFile::Object(obj) = file else {
                continue;
            };
            let Some(non_dynamic) = obj.non_dynamic.as_mut() else {
                continue;
            };

            for sec in &mut non_dynamic.merge_strings_sections {
                debug_assert_ne!(sec.part_id, part_id::CUSTOM_PLACEHOLDER);
                if sections
                    .insert((obj.file_id, sec.input_section_index), sec)
                    .is_some()
                {
                    bail!("Duplicate entry when partitioning merged-string sections")
                };
            }
        }
    }

    for builder in merge_builders.into_raw_values().into_iter() {
        for info in builder.string_to_info.into_read_only().values() {
            let file_id = info.file_id;

            let sec = sections
                .get_mut(&(file_id, info.section_index))
                .context("Internal error: Couldn't find where to put string-merge data")?;

            // If we get this wrong, we'll crash later anyway, but crashing here might be easier
            // to debug.
            debug_assert_bail!(
                info.range.end_offset as usize <= sec.section_data.len(),
                "Merge string {}..{} out of section {} range, ..{} in file {}",
                info.range.start_offset,
                info.range.end_offset,
                info.section_index,
                sec.section_data.len(),
                file_id,
            );

            sec.strings.push(MergeStringData {
                id: info.id,
                range: info.range,
            });
        }
    }
    Ok(())
}

/// Sorts merged strings by their start offset within the input section. This is necessary to ensure
/// deterministic output. We also compute the size of each files' contribution to each output
/// section.
#[tracing::instrument(skip_all, name = "Sort merged-string records")]
fn sort_merged_string_records<'data>(resolved: &mut [ResolvedGroup<'data>]) {
    resolved.par_iter_mut().for_each(|group| {
        for file in &mut group.files {
            let ResolvedFile::Object(obj) = file else {
                continue;
            };
            let Some(non_dynamic) = obj.non_dynamic.as_mut() else {
                continue;
            };

            for merge_info in &mut non_dynamic.merge_strings_sections {
                merge_info.strings.sort_by_key(|s| s.range.start_offset);
                merge_info.size = merge_info.strings.iter().map(|s| s.len() as u64).sum();
            }
        }
    })
}

impl MergeStringSectionSizes {
    fn new(merge_builders: &OutputSectionMap<MergeStringSectionBuilder>) -> Self {
        let sizes = merge_builders.map(|builder| builder.next_id.load(Ordering::Relaxed));
        Self { sizes }
    }

    #[tracing::instrument(skip_all, name = "Allocate merge string address storage")]
    pub(crate) fn allocate(&self) -> OutputSectionMap<Vec<AtomicU64>> {
        self.sizes.map(|size| {
            let mut v = Vec::new();
            v.resize_with(*size as usize, || AtomicU64::new(0));
            v
        })
    }
}

fn print_merge_metrics(
    merge_builders: &OutputSectionMap<MergeStringSectionBuilder>,
    output_sections: &OutputSections,
) {
    merge_builders.for_each(|section_id, builder| {
        let input_byte_count = builder.input_byte_count.load(Ordering::Relaxed);
        if input_byte_count > 0 {
            tracing::debug!(target: "metrics", section = ?output_sections.name(section_id),
            input_string_count = builder.input_string_count.load(Ordering::Relaxed),
            input_byte_count,
            output_string_count = builder.next_id.load(Ordering::Relaxed),
            output_byte_count = builder.output_byte_count(),
            "merge_strings");
        }
    });
}

impl<'data> MergeStringSectionBuilder<'data> {
    fn process_string(
        &self,
        string: PreHashed<StringToMerge<'data>>,
        file_id: FileId,
        section_index: object::SectionIndex,
        range: MergeStringRange,
    ) -> MergedStringId {
        self.input_byte_count
            .fetch_add(string.bytes.len() as u64, Ordering::Relaxed);
        self.input_string_count.fetch_add(1, Ordering::Relaxed);

        match self.string_to_info.entry(string) {
            dashmap::Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                // For now, we select the first file that defines a string as the winner. This will
                // result in less even sharding, since more string will be owned by earlier files,
                // however the theory is that it'll also result in more cache-friendly behaviour
                // when writing the output file, since we'll be copying strings that are consecutive
                // in memory more often.
                if file_id < existing.file_id {
                    existing.file_id = file_id;
                    existing.section_index = section_index;
                    existing.range = range;
                }
                existing.id
            }
            dashmap::Entry::Vacant(entry) => {
                let id = MergedStringId(self.next_id.fetch_add(1, Ordering::Relaxed));
                entry.insert(MergeStringInfo {
                    id,
                    file_id,
                    section_index,
                    range,
                });
                id
            }
        }
    }

    fn output_byte_count(&self) -> u64 {
        self.string_to_info
            .iter()
            .map(|info| info.range.len() as u64)
            .sum()
    }
}

impl MergeStringsSection {
    // DO NOT SUBMIT
}

impl<'data> StringToMerge<'data> {
    /// Takes from `source` up to the next null terminator. Returns a prehashed reference to what
    /// was taken.
    pub(crate) fn take_hashed(source: &mut &'data [u8]) -> Result<PreHashed<StringToMerge<'data>>> {
        let len = memchr::memchr(0, source)
            .map(|i| i + 1)
            .context("String in merge-string section is not null-terminated")?;
        let (bytes, rest) = source.split_at(len);
        let hash = crate::hash::hash_bytes(bytes);
        *source = rest;
        Ok(PreHashed::new(StringToMerge { bytes }, hash))
    }
}

/// Looks for a merged string at `symbol_index` + `addend` in the input and if found, returns its
/// address in the output.
pub(crate) fn get_merged_string_output_address(
    symbol_index: object::SymbolIndex,
    addend: u64,
    object: &crate::elf::File,
    sections: &[SectionSlot],
    merged_strings: &OutputSectionMap<MergeStringsSection>,
    merged_string_start_addresses: &MergedStringStartAddresses,
    merge_string_file_sections: &[MergeStringsFileSectionData],
    zero_unnamed: bool,
) -> Result<Option<u64>> {
    let symbol = object.symbol(symbol_index)?;
    let Some(section_index) = object.symbol_section(symbol, symbol_index)? else {
        return Ok(None);
    };
    let SectionSlot::MergeStrings(merge_slot) = &sections[section_index.0] else {
        return Ok(None);
    };
    let mut input_offset = symbol.st_value(LittleEndian);

    // When we reference data in a string-merge section via a named symbol, we determine which
    // string we're referencing without taking the addend into account, then apply the addend
    // afterward. However when the reference is to a section (a symbol without a name), we take the
    // addend into account up-front before we determine which string we're pointing at. This is a
    // bit weird, but seems to match what other linkers do.
    let symbol_has_name = symbol.st_name(LittleEndian) != 0;
    if !symbol_has_name {
        // We're computing a resolution for an unnamed symbol, just use the value of 0 for now.
        // We'll compute the address later when we're processing relocations that reference the
        // section.
        if zero_unnamed {
            return Ok(Some(0));
        }
        input_offset = input_offset.wrapping_add(addend);
    }

    let section_id = merge_slot.part_id.output_section_id();

    let file_section = &merge_string_file_sections[merge_slot.merge_info_index as usize];

    let string_id = file_section
        .offset_to_id
        .get(&input_offset)
        .with_context(|| {
            format!("Failed to find merge-string at offset {input_offset} in section {section_id}")
        })?;

    let mut address = merged_strings.get(section_id).id_to_offset[string_id.0 as usize];

    // DO NOT SUBMIT
    // let section_base = merged_string_start_addresses.addresses.get(section_id);
    // let mut address = section_base + output_offset;

    if symbol_has_name {
        address = address.wrapping_add(addend);
    }

    Ok(Some(address))
}

impl<'data> MergeStringsFileSectionData<'data> {
    pub(crate) fn new(section_data: &'data [u8], section_index: object::SectionIndex) -> Self {
        Self {
            input_section_index: section_index,
            part_id: part_id::CUSTOM_PLACEHOLDER,
            section_data,
            offset_to_id: Default::default(),
            strings: Default::default(),
            size: 0,
        }
    }

    pub(crate) fn assign_addresses(&self, next_address: &mut u64, addresses: &[AtomicU64]) {
        for string in &self.strings {
            addresses[string.id.0 as usize].store(*next_address, Ordering::Relaxed);
            *next_address += string.len() as u64;
        }
    }
}

impl MergedStringStartAddresses {
    #[tracing::instrument(skip_all, name = "Compute merged string section start addresses")]
    pub(crate) fn compute(
        output_sections: &OutputSections<'_>,
        starting_mem_offsets_by_group: &[OutputSectionPartMap<u64>],
    ) -> Self {
        let mut addresses = OutputSectionMap::with_size(output_sections.num_sections());
        let internal_start_offsets = starting_mem_offsets_by_group.first().unwrap();
        for i in 0..output_sections.num_regular_sections() {
            let section_id = OutputSectionId::regular(i as u32);
            *addresses.get_mut(section_id) =
                *internal_start_offsets.get(section_id.part_id_with_alignment(alignment::MIN));
        }
        Self { addresses }
    }
}

impl MergeStringRange {
    fn len(&self) -> usize {
        self.end_offset as usize - self.start_offset as usize
    }
}

impl MergeStringData {
    fn len(&self) -> usize {
        self.range.len()
    }

    pub(crate) fn usize_range(&self) -> Range<usize> {
        self.range.start_offset as usize..self.range.end_offset as usize
    }
}

impl std::fmt::Display for StringToMerge<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(self.bytes))
    }
}
