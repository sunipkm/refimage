//! [`Frame`] / [`ApplyInput`] — the input traits for [`Runner::run`] and the
//! one-shot [`Pipeline::apply`](super::Pipeline::apply).

use crate::{DynamicImageOwned, DynamicImageRef, GenericImageOwned, GenericImageRef, ImageProps};

use super::{ImageSpec, Pipeline, PipelineError, Runner, Strategy};

/// A frame the pipeline reads from: its shape (via [`ImageProps`]) and the pixel
/// data as native-endian bytes. Implemented for [`DynamicImageRef`] and
/// [`DynamicImageOwned`]; it is the input to [`Runner::run`] / [`Runner::run_into`]
/// and, through [`ApplyInput`], to [`Pipeline::apply`](super::Pipeline::apply).
pub trait Frame: ImageProps {
    /// The pixel data as a native-endian byte slice.
    fn as_bytes(&self) -> &[u8];
}

impl Frame for DynamicImageRef<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.as_raw_u8()
    }
}

impl Frame for DynamicImageOwned {
    fn as_bytes(&self) -> &[u8] {
        self.as_raw_u8()
    }
}

/// An image a [`Pipeline`] can be applied to: a [`DynamicImageRef`] /
/// [`DynamicImageOwned`], or a metadata-bearing [`GenericImageRef`] /
/// [`GenericImageOwned`]. [`Output`](ApplyInput::Output) is the corresponding owned
/// image type — a `Generic*` input keeps its metadata.
pub trait ApplyInput {
    /// The owned image [`Pipeline::apply`](super::Pipeline::apply) produces for this input.
    type Output;

    #[doc(hidden)]
    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError>;
}

/// Run `pipeline` once over `frame`, returning a fresh owned image.
fn apply_dynamic<F: Frame + ?Sized>(
    pipeline: &Pipeline,
    frame: &F,
) -> Result<DynamicImageOwned, PipelineError> {
    let mut runner: Runner =
        pipeline.compile(ImageSpec::from_dynamic(frame), Strategy::Sequential)?;
    let out = runner.run(frame)?;
    Ok(DynamicImageOwned::from(&out))
}

impl ApplyInput for DynamicImageRef<'_> {
    type Output = DynamicImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        apply_dynamic(pipeline, self)
    }
}

impl ApplyInput for DynamicImageOwned {
    type Output = DynamicImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        apply_dynamic(pipeline, self)
    }
}

impl ApplyInput for GenericImageRef<'_> {
    type Output = GenericImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        Ok(GenericImageOwned {
            metadata: self.metadata.clone(),
            image: apply_dynamic(pipeline, self.get_image())?,
        })
    }
}

impl ApplyInput for GenericImageOwned {
    type Output = GenericImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        Ok(GenericImageOwned {
            metadata: self.metadata.clone(),
            image: apply_dynamic(pipeline, self.get_image())?,
        })
    }
}
