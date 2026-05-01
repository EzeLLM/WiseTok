use crate::Pair;

/// A "word" in BPE training: a sequence of token IDs that started life as the
/// raw bytes of a single regex chunk. As training proceeds, adjacent IDs get
/// replaced with the new merged ID.
#[derive(Clone, Debug)]
pub(crate) struct Word {
    pub(crate) ids: Vec<u32>,
}

impl Word {
    #[inline]
    pub(crate) fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    pub(crate) fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of `pair` into `new_id`.
    /// Returns local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this deliberately avoids a HashMap in the hot loop.
    pub(crate) fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i64)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i64)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}
