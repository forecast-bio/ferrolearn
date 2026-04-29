//! Count vectorizer: convert text documents to a term-count matrix.
//!
//! Tokenizes documents by splitting on non-alphanumeric characters, builds a
//! vocabulary, and produces a term-count matrix of shape `(n_docs, n_vocab)`.

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// CountVectorizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted count vectorizer.
///
/// Tokenizes documents by splitting on non-alphanumeric boundaries, builds a
/// vocabulary sorted alphabetically, and transforms documents into a
/// term-count matrix.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::count_vectorizer::{CountVectorizer, FittedCountVectorizer};
///
/// let docs = vec![
///     "the cat sat".to_string(),
///     "the cat sat on the mat".to_string(),
/// ];
/// let cv = CountVectorizer::new();
/// let fitted = cv.fit(&docs).unwrap();
/// let counts = fitted.transform(&docs).unwrap();
/// assert_eq!(counts.nrows(), 2);
/// assert_eq!(counts.ncols(), fitted.vocabulary().len());
/// ```
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    /// Maximum number of features (vocabulary size). `None` means no limit.
    pub max_features: Option<usize>,
    /// Minimum document frequency (absolute count) for a term to be included.
    pub min_df: usize,
    /// Maximum document frequency as a fraction of total documents.
    /// Terms appearing in more than `max_df * n_docs` documents are excluded.
    pub max_df: f64,
    /// If `true`, all counts are clipped to 0/1 (binary occurrence).
    pub binary: bool,
    /// If `true`, lowercase all tokens before counting.
    pub lowercase: bool,
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            binary: false,
            lowercase: true,
        }
    }

    /// Set the maximum number of features.
    #[must_use]
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set the minimum document frequency.
    #[must_use]
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set the maximum document frequency as a fraction of total documents.
    #[must_use]
    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Enable or disable binary mode.
    #[must_use]
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Enable or disable lowercasing.
    #[must_use]
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Fit the vectorizer on a corpus of documents.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the corpus is empty.
    /// Returns [`FerroError::InvalidParameter`] if `max_df` is not in `(0, 1]`.
    pub fn fit(&self, docs: &[String]) -> Result<FittedCountVectorizer, FerroError> {
        let n_docs = docs.len();
        if n_docs == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "CountVectorizer::fit".into(),
            });
        }
        if self.max_df <= 0.0 || self.max_df > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_df".into(),
                reason: format!("must be in (0, 1], got {}", self.max_df),
            });
        }

        // Build document-frequency counts.
        let mut df_counts: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            let tokens = tokenize(doc, self.lowercase);
            // Unique tokens per document.
            let mut seen = std::collections::HashSet::new();
            for tok in tokens {
                if seen.insert(tok.clone()) {
                    *df_counts.entry(tok).or_insert(0) += 1;
                }
            }
        }

        // Filter by min_df and max_df.
        let max_df_abs = (self.max_df * n_docs as f64).ceil() as usize;
        let mut vocab: Vec<String> = df_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_abs)
            .map(|(term, _)| term)
            .collect();
        vocab.sort();

        // Apply max_features: keep the top-N by total corpus frequency.
        if let Some(max_f) = self.max_features {
            if vocab.len() > max_f {
                // Re-count total frequencies for the remaining terms.
                let mut total_freq: HashMap<String, usize> = HashMap::new();
                for doc in docs {
                    let tokens = tokenize(doc, self.lowercase);
                    for tok in tokens {
                        if vocab.binary_search(&tok).is_ok() {
                            *total_freq.entry(tok).or_insert(0) += 1;
                        }
                    }
                }
                // Sort by descending frequency, then alphabetically for ties.
                vocab.sort_by(|a, b| {
                    let fa = total_freq.get(a).unwrap_or(&0);
                    let fb = total_freq.get(b).unwrap_or(&0);
                    fb.cmp(fa).then_with(|| a.cmp(b))
                });
                vocab.truncate(max_f);
                vocab.sort(); // restore alphabetical order for consistent indexing
            }
        }

        // Build vocabulary mapping.
        let vocabulary: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        Ok(FittedCountVectorizer {
            vocabulary,
            sorted_terms: vocab,
            binary: self.binary,
            lowercase: self.lowercase,
        })
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedCountVectorizer
// ---------------------------------------------------------------------------

/// A fitted count vectorizer holding the learned vocabulary.
///
/// Created by calling [`CountVectorizer::fit`].
#[derive(Debug, Clone)]
pub struct FittedCountVectorizer {
    /// Map from term to column index.
    vocabulary: HashMap<String, usize>,
    /// Sorted vocabulary terms (for deterministic column ordering).
    sorted_terms: Vec<String>,
    /// Whether to clip counts to binary.
    binary: bool,
    /// Whether to lowercase tokens.
    lowercase: bool,
}

impl FittedCountVectorizer {
    /// Return the vocabulary as a sorted slice of terms.
    #[must_use]
    pub fn vocabulary(&self) -> &[String] {
        &self.sorted_terms
    }

    /// Return the vocabulary mapping (term -> column index).
    #[must_use]
    pub fn vocabulary_map(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Transform documents into a term-count matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if `docs` is empty.
    pub fn transform(&self, docs: &[String]) -> Result<Array2<f64>, FerroError> {
        if docs.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedCountVectorizer::transform".into(),
            });
        }

        let n_docs = docs.len();
        let n_vocab = self.sorted_terms.len();
        let mut matrix = Array2::<f64>::zeros((n_docs, n_vocab));

        for (i, doc) in docs.iter().enumerate() {
            let tokens = tokenize(doc, self.lowercase);
            for tok in tokens {
                if let Some(&col) = self.vocabulary.get(&tok) {
                    if self.binary {
                        matrix[[i, col]] = 1.0;
                    } else {
                        matrix[[i, col]] += 1.0;
                    }
                }
            }
        }

        Ok(matrix)
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Tokenize a document by splitting on non-alphanumeric boundaries.
fn tokenize(doc: &str, lowercase: bool) -> Vec<String> {
    let text = if lowercase {
        doc.to_lowercase()
    } else {
        doc.to_string()
    };

    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = vec![
            "the cat sat".to_string(),
            "the cat sat on the mat".to_string(),
        ];
        let cv = CountVectorizer::new();
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();

        assert_eq!(counts.nrows(), 2);
        let vocab = fitted.vocabulary();
        assert!(vocab.contains(&"cat".to_string()));
        assert!(vocab.contains(&"the".to_string()));
        assert!(vocab.contains(&"sat".to_string()));

        // "the" appears once in doc 0, twice in doc 1
        let the_idx = fitted.vocabulary_map()["the"];
        assert_abs_diff_eq!(counts[[0, the_idx]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(counts[[1, the_idx]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_binary() {
        let docs = vec!["the the the".to_string()];
        let cv = CountVectorizer::new().binary(true);
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();
        // "the" count should be 1 (binary mode)
        assert_abs_diff_eq!(counts[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_lowercase() {
        let docs = vec!["Hello HELLO hello".to_string()];
        let cv = CountVectorizer::new();
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();
        // All should fold to "hello", count = 3
        assert_eq!(fitted.vocabulary().len(), 1);
        assert_abs_diff_eq!(counts[[0, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_no_lowercase() {
        let docs = vec!["Hello hello".to_string()];
        let cv = CountVectorizer::new().lowercase(false);
        let fitted = cv.fit(&docs).unwrap();
        // "Hello" and "hello" are different tokens
        assert_eq!(fitted.vocabulary().len(), 2);
    }

    #[test]
    fn test_count_vectorizer_max_features() {
        let docs = vec!["a b c d e f".to_string()];
        let cv = CountVectorizer::new().max_features(3);
        let fitted = cv.fit(&docs).unwrap();
        assert_eq!(fitted.vocabulary().len(), 3);
    }

    #[test]
    fn test_count_vectorizer_min_df() {
        let docs = vec![
            "cat dog".to_string(),
            "cat bird".to_string(),
            "cat fish".to_string(),
        ];
        // Only "cat" appears in all 3 docs
        let cv = CountVectorizer::new().min_df(3);
        let fitted = cv.fit(&docs).unwrap();
        assert_eq!(fitted.vocabulary().len(), 1);
        assert_eq!(fitted.vocabulary()[0], "cat");
    }

    #[test]
    fn test_count_vectorizer_max_df() {
        let docs = vec![
            "the cat".to_string(),
            "the dog".to_string(),
            "the bird".to_string(),
        ];
        // "the" appears in 100% of docs. max_df=0.5 should exclude it.
        let cv = CountVectorizer::new().max_df(0.5);
        let fitted = cv.fit(&docs).unwrap();
        assert!(!fitted.vocabulary().contains(&"the".to_string()));
    }

    #[test]
    fn test_count_vectorizer_empty_corpus() {
        let docs: Vec<String> = vec![];
        let cv = CountVectorizer::new();
        assert!(cv.fit(&docs).is_err());
    }

    #[test]
    fn test_count_vectorizer_transform_empty() {
        let docs = vec!["hello world".to_string()];
        let fitted = CountVectorizer::new().fit(&docs).unwrap();
        let empty: Vec<String> = vec![];
        assert!(fitted.transform(&empty).is_err());
    }

    #[test]
    fn test_count_vectorizer_unseen_tokens() {
        let train = vec!["cat dog".to_string()];
        let fitted = CountVectorizer::new().fit(&train).unwrap();
        let test = vec!["fish bird".to_string()];
        let counts = fitted.transform(&test).unwrap();
        // All zeros since no tokens match
        for &v in &counts {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }
}
