use {
    ndarray::ShapeError,
    ort::Error as OrtError,
    std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
        io::Error as IoError,
        num::ParseIntError,
        time::SystemTimeError,
    },
};

#[derive(Debug)]
pub enum TranslationError {
    Io(IoError),
    Ort(OrtError),
    ParseInt(ParseIntError),
    Shape(ShapeError),
    SystemTime(SystemTimeError),
    WordTokenNotFound(String),
    WordIdTokenNotFound(i64),
}

impl Clone for TranslationError {
    fn clone(&self) -> Self {
        match self {
            Self::Io(e) => Self::Io(IoError::new(e.kind(), e.to_string())),
            Self::Ort(e) => Self::Ort(OrtError::new(e.message())),
            Self::ParseInt(e) => Self::ParseInt(e.clone()),
            Self::Shape(e) => Self::Shape(e.clone()),
            Self::SystemTime(e) => Self::SystemTime(e.clone()),
            Self::WordTokenNotFound(e) => Self::WordTokenNotFound(e.to_owned()),
            Self::WordIdTokenNotFound(e) => Self::WordIdTokenNotFound(*e),
        }
    }
}

impl Display for TranslationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "TranslationError: ")?;
        match self {
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(e) => Display::fmt(e, f),
            Self::ParseInt(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
            Self::SystemTime(e) => Display::fmt(e, f),
            Self::WordTokenNotFound(w) => {
                write!(f, "WordTokenNotFound: the word `{w}` is not in vocab.")
            }
            Self::WordIdTokenNotFound(id) => {
                write!(
                    f,
                    "WordIdTokenNotFound: the word id `{id}` is not in vocab."
                )
            }
        }
    }
}

impl Error for TranslationError {}

impl From<ShapeError> for TranslationError {
    fn from(value: ShapeError) -> Self {
        Self::Shape(value)
    }
}

impl From<OrtError> for TranslationError {
    fn from(value: OrtError) -> Self {
        Self::Ort(value)
    }
}

impl From<SystemTimeError> for TranslationError {
    fn from(value: SystemTimeError) -> Self {
        Self::SystemTime(value)
    }
}

impl From<IoError> for TranslationError {
    fn from(value: IoError) -> Self {
        Self::Io(value)
    }
}

impl From<ParseIntError> for TranslationError {
    fn from(value: ParseIntError) -> Self {
        Self::ParseInt(value)
    }
}
