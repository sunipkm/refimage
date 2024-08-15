use crate::Primitive;

/// Enum to hold the data store.
#[derive(Debug, PartialEq)]
enum DataStorEnum<'a, T: Primitive> {
    /// A reference to a slice of data.
    Ref(&'a mut [T]),
    /// Owned data.
    Own(Vec<T>),
}

/// Private struct to hold the data store.
#[derive(Debug, PartialEq)]
pub struct DataStor<'a, T: Primitive>(DataStorEnum<'a, T>);

impl<'a, T: Primitive> DataStor<'a, T> {
    /// Create a new data store from owned data.
    pub fn from_mut_ref(data: &'a mut [T]) -> Self {
        DataStor(DataStorEnum::Ref(data))
    }

    /// Create a new data store from owned data.
    pub fn from_owned(data: Vec<T>) -> Self {
        DataStor(DataStorEnum::Own(data))
    }

    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        match &self.0 {
            DataStorEnum::Ref(data) => data,
            DataStorEnum::Own(data) => data.as_slice(),
        }
    }

    /// Get the data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.0 {
            DataStorEnum::Ref(data) => data,
            DataStorEnum::Own(data) => data,
        }
    }

    /// Get the data as a vector.
    pub fn into_vec(self) -> Vec<T> {
        match self.0 {
            DataStorEnum::Own(data) => data,
            DataStorEnum::Ref(data) => data.to_vec(),
        }
    }

    /// Get the raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        match &self.0 {
            DataStorEnum::Ref(data) => data.as_ptr(),
            DataStorEnum::Own(data) => data.as_ptr(),
        }
    }

    /// Get the raw mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match &mut self.0 {
            DataStorEnum::Ref(data) => data.as_mut_ptr(),
            DataStorEnum::Own(data) => data.as_mut_ptr(),
        }
    }

    /// Get an iterator over the data.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.as_slice().iter()
    }

    /// Get a mutable iterator over the data.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.as_mut_slice().iter_mut()
    }

    /// Get a u8 slice of the data.
    /// # Safety
    /// This function is unsafe because it returns a slice of u8.
    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }

    /// Safely get a u8 slice of the data.
    pub fn as_u8_slice_checked(&self) -> Option<&[u8]> {
        bytemuck::try_cast_slice(self.as_slice()).ok()
    }

    /// Get the length of the data.
    pub fn len(&self) -> usize {
        match &self.0 {
            DataStorEnum::Ref(data) => data.len(),
            DataStorEnum::Own(data) => data.len(),
        }
    }

    /// Whether the data is empty.
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            DataStorEnum::Ref(data) => data.is_empty(),
            DataStorEnum::Own(data) => data.is_empty(),
        }
    }
}

impl<'a, T: Primitive> DataStor<'a, T> {
    /// Convert to owned data.
    pub fn to_owned(&self) -> Self {
        self.clone()
    }
}

impl<'a, T: Primitive> Clone for DataStor<'a, T> {
    fn clone(&self) -> Self {
        match &self.0 {
            DataStorEnum::Ref(data) => DataStor(DataStorEnum::Own(data.to_vec())),
            DataStorEnum::Own(data) => DataStor(DataStorEnum::Own(data.clone())),
        }
    }
}
