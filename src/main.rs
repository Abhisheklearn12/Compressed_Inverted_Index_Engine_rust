// Features:
// - Tokenization (simple whitespace + punctuation normalization)
// - In-memory index builder: mapping term -> postings list
// - Postings stored as sorted docIDs, delta-encoded and varint-compressed (VByte)
// - Basic query engine: AND (conjunction) and OR (disjunction) over terms
// - On-disk serialization/deserialization (simple block format)
// - Unit tests and example usage
//
// Notes on compression:
// - We delta-encode document IDs in a postings list (store gaps), which reduces magnitude
//   of values when docIDs are relatively close.
// - We use a simple variable-byte (vbyte) encoding (also called "VByte") to store
//   unsigned integers. This is easy to implement and decode quickly.
// - For production, consider using LEB128, Simple9/16, or PForDelta for better speed/space.
//
// Correctness guarantees:
// - The implementation carefully encodes/decodes integers and uses stable sorting to
//   ensure deterministic index output.
// - Query operations are implemented using merge algorithms on decoded postings to
//   preserve correctness.
//
// Limitations (by design for clarity):
// - Tokenization is intentionally simple (lowercase, split on non-alphanumeric).
// - No term positions (so phrase queries are not supported).
// - This is a learning-friendly implementation, not optimized for extreme scale.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

// ----------------------------
// Utilities: VByte encode/decode
// ----------------------------

/// Encode a u32 into VByte (variable byte) form and append to a vector.
/// VByte uses the MSB (0x80) to mark the final byte of a number.
fn vbyte_encode(mut value: u32, out: &mut Vec<u8>) {
    // Emit 7 bits per byte, low-order first
    let mut bytes = [0u8; 5];
    let mut idx = 0;
    loop {
        let b = (value & 0x7F) as u8;
        bytes[idx] = b;
        idx += 1;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    // write bytes with continuation bit set except last
    for i in 0..(idx - 1) {
        out.push(bytes[i] | 0x80);
    }
    out.push(bytes[idx - 1]);
}

/// Decode a single u32 from VByte encoded slice starting at `pos`.
/// Returns (value, bytes_read).
fn vbyte_decode(buf: &[u8], pos: usize) -> Option<(u32, usize)> {
    let mut shift = 0u32;
    let mut result: u32 = 0;
    let mut i = pos;
    while i < buf.len() {
        let byte = buf[i];
        let part = (byte & 0x7F) as u32;
        result |= part << shift;
        i += 1;
        if (byte & 0x80) == 0 {
            return Some((result, i - pos));
        }
        shift += 7;
        if shift >= 35 {
            // overflow for u32
            return None;
        }
    }
    None
}

// ----------------------------
// Tokenizer
// ----------------------------

/// Very simple tokenizer: lowercase + split on non-alphanumeric.
fn tokenize(text: &str) -> Vec<String> {
    let mut tok = String::new();
    let mut out = Vec::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            tok.push(ch.to_ascii_lowercase());
        } else {
            if !tok.is_empty() {
                out.push(tok.clone());
                tok.clear();
            }
        }
    }
    if !tok.is_empty() {
        out.push(tok);
    }
    out
}

// ----------------------------
// Postings and compression helpers
// ----------------------------

/// Delta-encode a sorted slice of docIDs (u32). Returns gaps (first is original first value).
fn delta_encode(sorted: &[u32]) -> Vec<u32> {
    let mut res = Vec::with_capacity(sorted.len());
    let mut prev = 0u32;
    for (i, &d) in sorted.iter().enumerate() {
        if i == 0 {
            res.push(d);
        } else {
            res.push(d - prev);
        }
        prev = d;
    }
    res
}

/// Delta-decode: given gaps, reconstruct original docIDs.
fn delta_decode(gaps: &[u32]) -> Vec<u32> {
    let mut res = Vec::with_capacity(gaps.len());
    let mut acc = 0u32;
    for &g in gaps.iter() {
        acc = acc.wrapping_add(g);
        res.push(acc);
    }
    res
}

/// Compress a postings list (sorted docIDs) into bytes using delta + vbyte.
fn compress_postings(sorted_docids: &[u32]) -> Vec<u8> {
    let gaps = delta_encode(sorted_docids);
    let mut out = Vec::new();
    // store number of postings first as vbyte
    vbyte_encode(gaps.len() as u32, &mut out);
    for g in gaps.iter() {
        vbyte_encode(*g, &mut out);
    }
    out
}

/// Decompress postings from a byte buffer (starting at 0). Returns docIDs.
fn decompress_postings(buf: &[u8]) -> io::Result<Vec<u32>> {
    // read length
    let (len, mut pos) = vbyte_decode(buf, 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "vbyte decode failed (len)"))?;
    let mut gaps = Vec::with_capacity(len as usize);
    for _ in 0..len {
        let (v, read) = vbyte_decode(buf, pos).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "vbyte decode failed (gap)")
        })?;
        gaps.push(v);
        pos += read;
    }
    Ok(delta_decode(&gaps))
}

// ----------------------------
// Inverted Index Structures
// ----------------------------

/// On-disk format (simple):
/// [header]
/// term_count: u32 (vbyte)
/// for each term (sorted by term string):
///   term_len: u32 (vbyte)
///   term_bytes: [u8]
///   postings_offset: u64 (8 bytes, little endian)
/// After term directory, the postings data is appended. Each posting list is stored as:
///   compressed bytes as produced by compress_postings.
/// The term directory allows seeking to postings offsets without reading everything.
///
/// In-memory representation while building: map term -> Vec<u32> (docIDs sorted)
#[derive(Debug, Default)]
pub struct IndexBuilder {
    // Using BTreeMap to have deterministic ordering when serializing
    terms: BTreeMap<String, Vec<u32>>,
    next_doc_id: u32,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self {
            terms: BTreeMap::new(),
            next_doc_id: 1, // start docIDs at 1 for clarity
        }
    }

    /// Add a document; returns the assigned docID.
    pub fn add_document(&mut self, text: &str) -> u32 {
        let doc_id = self.next_doc_id;
        self.next_doc_id = self.next_doc_id.saturating_add(1);
        let tokens = tokenize(text);
        let mut seen = std::collections::HashSet::new();
        for t in tokens.into_iter() {
            // simple term deduplication per doc (posting lists typically store unique docIDs)
            if seen.insert(t.clone()) {
                self.terms.entry(t).or_default().push(doc_id);
            }
        }
        doc_id
    }

    /// Finalize and produce a `CompressedIndex` (which is read/query-able and serializable).
    pub fn finalize(mut self) -> CompressedIndex {
        // Ensure each postings list is sorted and unique
        for (_term, list) in self.terms.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        CompressedIndex::from_terms(self.terms)
    }
}

/// CompressedIndex holds compressed postings in memory (as bytes) and a term -> offset directory.
pub struct CompressedIndex {
    // Term -> (offset, length) directory. We use BTreeMap to keep terms sorted for stable dumps.
    pub directory: BTreeMap<String, (u64, u64)>,
    pub postings_blob: Vec<u8>,
}

impl CompressedIndex {
    /// Build from already sorted & deduped term -> postings
    pub fn from_terms(terms: BTreeMap<String, Vec<u32>>) -> Self {
        let mut blob = Vec::new();
        let mut dir = BTreeMap::new();
        // we'll append postings and record offsets
        for (term, postings) in terms.into_iter() {
            let offset = blob.len() as u64;
            let compressed = compress_postings(&postings);
            let len = compressed.len() as u64;
            blob.extend_from_slice(&compressed);
            dir.insert(term, (offset, len));
        }
        Self {
            directory: dir,
            postings_blob: blob,
        }
    }

    /// Save to file
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut f = File::create(path)?;
        self.write_to(&mut f)
    }

    /// Write to any Write+Seek (file-like)
    pub fn write_to<W: Write + Seek>(&self, w: &mut W) -> io::Result<()> {
        // write term count as vbyte
        vbyte_encode(self.directory.len() as u32, &mut Vec::new()); // no-op but kept for symmetry
        // We'll write to a temporary buffer then flush to w
        // Header: term_count (vbyte), then for each term: term_len (vbyte), term_bytes, postings_offset(u64)
        let mut header = Vec::new();
        vbyte_encode(self.directory.len() as u32, &mut header);
        // We'll write postings after header, but we need to reserve space for offsets.
        // We'll write term entries with placeholder offsets, then write postings and come back to fill offsets.
        // To simplify, compute postings start offset: header_size + sum(term entries sizes)

        // First compute term entries sizes, collecting term bytes
        let mut term_entries: Vec<(String, Vec<u8>)> = Vec::new();
        for (term, _) in self.directory.iter() {
            term_entries.push((term.clone(), term.as_bytes().to_vec()));
        }

        // term_count already in header; now append term entries without offsets
        for (_term, bytes) in term_entries.iter() {
            vbyte_encode(bytes.len() as u32, &mut header);
            header.extend_from_slice(bytes);
            // placeholder 8 bytes for u64 offset
            header.extend_from_slice(&[0u8; 8]);
        }

        // write header to file
        w.write_all(&header)?;

        // Now postings start at current file position
        let postings_start = w.seek(SeekFrom::Current(0))?;

        // Write postings in the same order as directory iteration
        for (_term, &(offset, len)) in self.directory.iter() {
            // offset is relative to postings_start; but our stored offset will be absolute from start of postings blob
            let postings_bytes = &self.postings_blob[offset as usize..(offset + len) as usize];
            w.write_all(postings_bytes)?;
        }

        // Now go back and write the offsets
        let mut cursor = HeaderCursor::new();
        // skip term_count vbyte
        let (_, rc) = vbyte_decode(&header, 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "header vbyte corrupt"))?;
        cursor.pos = rc as u64;
        // iterate and patch offsets on disk
        for (_term, &(offset, _len)) in self.directory.iter() {
            // read term_len vbyte
            let (tlen, read) = vbyte_decode(&header, cursor.pos as usize)
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "header parse error"))?;
            cursor.pos += read as u64;
            // skip term bytes
            cursor.pos += tlen as u64;
            // now current position is where placeholder offset was written in header; compute absolute offset of postings for this term in file
            // postings_offset_in_file = postings_start + offset
            let file_offset_to_write = postings_start + offset;
            // seek to header offset location in file and write u64 LE
            let where_to_write = (cursor.pos) as i64; // it's header absolute offset from file start
            w.seek(SeekFrom::Start(where_to_write as u64))?;
            w.write_all(&file_offset_to_write.to_le_bytes())?;
            // restore position to end of header writing
            // actually our header write finished earlier so we just continue.
            // advance cursor past the u64 we just wrote
            cursor.pos += 8;
        }
        Ok(())
    }

    /// Load index from a file
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut f = File::open(path)?;
        Self::read_from(&mut f)
    }

    /// Read from any Read+Seek
    pub fn read_from<R: Read + Seek>(r: &mut R) -> io::Result<Self> {
        // Read entire file into memory (for simplicity). For huge indexes, use streaming.
        let mut data = Vec::new();
        r.seek(SeekFrom::Start(0))?;
        r.read_to_end(&mut data)?;
        // parse header
        let (term_count, mut pos) = vbyte_decode(&data, 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "header parse error"))?;
        let mut dir = BTreeMap::new();
        // we'll extract term entries and offsets
        for _ in 0..term_count {
            let (tlen, read) = vbyte_decode(&data, pos)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "term len parse"))?;
            pos += read;
            let tlen_usize = tlen as usize;
            if pos + tlen_usize > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "term bytes out of range",
                ));
            }
            let term_bytes = &data[pos..pos + tlen_usize];
            let term = String::from_utf8(term_bytes.to_vec())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "term not utf8"))?;
            pos += tlen_usize;
            if pos + 8 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "offset out of range",
                ));
            }
            let offset_le = &data[pos..pos + 8];
            let offset = u64::from_le_bytes(offset_le.try_into().unwrap());
            pos += 8;
            // We'll compute length later by inspecting next posting's offset or EOF
            dir.insert(term, (offset, 0));
        }
        // compute lengths by ordering
        let mut pairs: Vec<(String, u64)> =
            dir.iter().map(|(t, &(off, _))| (t.clone(), off)).collect();
        pairs.sort_by_key(|(_, off)| *off);
        let mut result_dir = BTreeMap::new();
        for i in 0..pairs.len() {
            let term = &pairs[i].0;
            let off = pairs[i].1;
            let len = if i + 1 < pairs.len() {
                pairs[i + 1].1 - off
            } else {
                data.len() as u64 - off
            };
            result_dir.insert(term.clone(), (off, len));
        }
        // extract blob as the bytes from min offset to EOF
        let min_off = result_dir
            .values()
            .map(|(o, _)| *o)
            .min()
            .unwrap_or(data.len() as u64);
        let min_off_usize = min_off as usize;
        let blob = data[min_off_usize..].to_vec();
        // adjust offsets to be relative to blob start
        let adjusted_dir = result_dir
            .into_iter()
            .map(|(t, (o, l))| (t, (o - min_off, l)))
            .collect();
        Ok(Self {
            directory: adjusted_dir,
            postings_blob: blob,
        })
    }

    /// Retrieve postings (docID list) for a term. Returns sorted docIDs.
    pub fn get_postings(&self, term: &str) -> io::Result<Vec<u32>> {
        if let Some(&(off, len)) = self.directory.get(term) {
            let start = off as usize;
            let end = (off + len) as usize;
            if end > self.postings_blob.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "postings out of range",
                ));
            }
            decompress_postings(&self.postings_blob[start..end])
        } else {
            Ok(Vec::new())
        }
    }

    /// Boolean AND of multiple terms (conjunction). Returns docIDs that contain all terms.
    pub fn query_and(&self, terms: &[&str]) -> io::Result<Vec<u32>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        // fetch postings for each term
        let mut lists: Vec<Vec<u32>> = Vec::with_capacity(terms.len());
        for &t in terms.iter() {
            lists.push(self.get_postings(t)?);
        }
        // sort lists by length ascending to optimize intersection
        lists.sort_by_key(|l| l.len());
        // intersect iteratively
        let mut iter = lists.iter();
        let mut result = iter.next().cloned().unwrap_or_default();
        for lst in iter {
            result = intersect_sorted(&result, lst);
        }
        Ok(result)
    }

    /// Boolean OR of multiple terms (disjunction). Returns sorted unique docIDs.
    pub fn query_or(&self, terms: &[&str]) -> io::Result<Vec<u32>> {
        let mut acc: Vec<u32> = Vec::new();
        for &t in terms.iter() {
            let p = self.get_postings(t)?;
            acc = union_sorted(&acc, &p);
        }
        Ok(acc)
    }
}

// small helper to track header cursor position when writing offsets
struct HeaderCursor {
    pos: u64,
}
impl HeaderCursor {
    fn new() -> Self {
        Self { pos: 0 }
    }
}

// ----------------------------
// Merging utilities for queries
// ----------------------------

/// Intersect two sorted vectors of u32
fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut i = 0;
    let mut j = 0;
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Union of two sorted vectors (unique)
fn union_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut i = 0;
    let mut j = 0;
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    while i < a.len() {
        out.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        out.push(b[j]);
        j += 1;
    }
    out
}

// ----------------------------
// Examples and tests
// ----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_vbyte_roundtrip() {
        let mut out = Vec::new();
        let vals = [0u32, 1, 127, 128, 129, 16384, u32::MAX / 2];
        for v in vals.iter() {
            vbyte_encode(*v, &mut out);
        }
        // decode sequentially
        let mut pos = 0usize;
        for &expected in vals.iter() {
            let (v, read) = vbyte_decode(&out, pos).expect("decode");
            assert_eq!(v, expected);
            pos += read;
        }
    }

    #[test]
    fn test_delta_roundtrip() {
        let docs = vec![1u32, 5, 9, 10, 1000];
        let c = compress_postings(&docs);
        let decoded = decompress_postings(&c).unwrap();
        assert_eq!(decoded, docs);
    }

    #[test]
    fn test_index_build_and_query() {
        let mut b = IndexBuilder::new();
        b.add_document("The quick brown fox jumps over the lazy dog."); // doc 1
        b.add_document("Never jump over the lazy dog quickly."); // doc 2
        b.add_document("foxes are quick and clever."); // doc 3
        let idx = b.finalize();
        // check postings for "quick"
        let q = idx.get_postings("quick").unwrap();
        assert_eq!(q, vec![1, 3]);
        // AND query
        let andres = idx.query_and(&["quick", "fox"]).unwrap();
        assert_eq!(andres, vec![1]);
        // OR query
        let orres = idx.query_or(&["lazy", "clever"]).unwrap();
        assert_eq!(orres, vec![1, 2, 3]);
    }

    #[test]
    fn test_serialize_and_load() {
        let mut b = IndexBuilder::new();
        b.add_document("Alpha beta gamma delta");
        b.add_document("Beta gamma epsilon");
        b.add_document("Gamma zeta alpha");
        let idx = b.finalize();
        let path = "test_index.bin";
        idx.save_to_path(path).unwrap();
        let loaded = CompressedIndex::load_from_path(path).unwrap();
        // compare postings for terms
        for term in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"].iter() {
            let a = idx.get_postings(term).unwrap();
            let b = loaded.get_postings(term).unwrap();
            assert_eq!(a, b);
        }
        fs::remove_file(path).ok();
    }
}

// ----------------------------
// Example: small CLI to build and query (for manual testing).
// Not exposed as a separate binary; keep as demonstration code.
// ----------------------------

#[allow(dead_code)]
fn example_usage() -> io::Result<()> {
    // Build index
    let mut b = IndexBuilder::new();
    b.add_document("The quick brown fox jumps over the lazy dog");
    b.add_document("The quick red fox jumped over the sleeping cat");
    let idx = b.finalize();
    // Query
    let res = idx.query_and(&["quick", "fox"])?;
    println!("AND quick AND fox -> {:?}", res);
    // Save
    idx.save_to_path("example.idx")?;
    // Load
    let loaded = CompressedIndex::load_from_path("example.idx")?;
    let res2 = loaded.query_or(&["dog", "cat"])?;
    println!("OR dog OR cat -> {:?}", res2);
    Ok(())
}

fn main() -> std::io::Result<()> {
    example_usage()
}
